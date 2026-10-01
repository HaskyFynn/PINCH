"""A single inference owner keeps models and tracker state off the GUI thread."""
import queue
import threading
import time
import traceback
import numpy as np
from .registry import Registry
from .enrollment import Enrollment, VIEWS
from .pipeline import Pipeline, Observation
from .source import CameraSource, VideoSource
from .session import Session
from .vision import Models, crop_quality


class Worker(threading.Thread):
    def __init__(self, settings):
        super().__init__(name='pinch-inference', daemon=True)
        self.settings = settings
        self.commands = queue.Queue()
        self.events = queue.Queue(maxsize=128)
        self.frames = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.source = None
        self.source_description = ''
        self.source_mode = ''
        self.mode = 'preview'
        self.session = None
        self.enrollment = None
        self.pending_profile = None

    def send(self, command, **args):
        self.commands.put((command, args))

    def emit(self, event, **data):
        item = {'event':event, **data}
        try:
            self.events.put_nowait(item)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(item)

    def stop_session(self):
        if self.session:
            self.session.close()
            self.emit('status', message=f'Session saved: {self.session.directory}')
            self.session = None
            if self.mode == 'run':
                self.mode = 'preview'
                self.emit('mode', mode=self.mode)

    def switch_registry(self, registry):
        # Saving occurs before replacing the in-memory state.
        registry.save(self.settings.path('registry'))
        self.registry = registry
        self.pipeline.reset(registry)
        self.emit('registry', names=registry.names(), warnings=registry.warnings(self.models.embedder_hash))

    def handle(self, command, args):
        if command == 'source':
            self.stop_session()
            self.mode, self.enrollment, self.pending_profile = 'preview', None, None
            if self.source:
                self.source.close()
                self.source = None
            self.pipeline.reset()
            self.source_mode = args['kind']
            self.source_description = str(args['value'])
            self.source = (CameraSource(int(args['value']), self.settings) if args['kind']=='camera'
                           else VideoSource(args['value']))
            self.emit('mode', mode='preview')
        elif command == 'run':
            if not self.source:
                raise ValueError('Open a camera or video first')
            if self.session:
                return
            if self.source_mode == 'video':
                self.source.close()
                self.source = VideoSource(self.source_description)
            self.enrollment, self.pending_profile = None, None
            self.pipeline.reset()
            self.session = Session(self.settings, self.models, self.registry, self.source_description,
                                   record_video=args.get('record_video',False))
            self.mode = 'run'
            self.emit('mode', mode=self.mode)
        elif command == 'stop':
            self.stop_session()
            self.mode, self.enrollment, self.pending_profile = 'preview', None, None
            self.pipeline.reset()
            self.emit('mode', mode=self.mode)
        elif command == 'enroll':
            if not self.source:
                raise ValueError('Open a camera or video first')
            name = args['name'].strip()
            if not name or name.lower() == 'unknown' or len(name)>64:
                raise ValueError('Enter a marker name of 1–64 characters other than "unknown"')
            self.stop_session()
            if self.source_mode == 'video':
                self.source.close()
                self.source = VideoSource(self.source_description)
            self.enrollment = Enrollment(name, self.settings.enrollment_samples_per_view,
                                         self.settings.enrollment_interval)
            self.pending_profile = None
            self.mode = 'enroll'
            self.emit('mode', mode=self.mode)
        elif command == 'next_view':
            if self.enrollment:
                self.enrollment.advance()
        elif command == 'save_enrollment':
            if self.enrollment and self.enrollment.complete:
                profile = self.enrollment.build(self.models.embedder_hash, self.source_mode, self.source_description)
                # The GUI explicitly confirms replacement and close competing profiles.
                conflicts = []
                p = np.asarray(profile.proto)
                for m in self.registry.markers:
                    if m.marker_id != profile.marker_id:
                        similarity = float(np.max(p @ np.asarray(m.proto).T))
                        if similarity >= min(profile.thr, m.thr)-self.settings.identity_margin:
                            conflicts.append(m.marker_id)
                self.pending_profile = profile
                self.emit('confirm_profile', name=profile.marker_id, replace=profile.marker_id in self.registry.names(),
                          conflicts=conflicts, threshold=profile.thr)
            else:
                raise ValueError('Finish all five enrollment views first')
        elif command == 'commit_profile':
            if self.pending_profile:
                self.switch_registry(self.registry.with_profile(self.pending_profile))
                self.emit('status', message=f'Saved {self.pending_profile.marker_id}. Test it alongside the other markers.')
                self.enrollment, self.pending_profile, self.mode = None, None, 'preview'
                self.emit('mode', mode=self.mode)
        elif command == 'import':
            replacement = Registry.load(args['path'])
            self.stop_session()
            self.switch_registry(replacement)
            self.enrollment, self.pending_profile, self.mode = None, None, 'preview'
            self.emit('mode', mode=self.mode)
        elif command == 'reload':
            registry = Registry.load(self.settings.path('registry'))
            self.stop_session()
            self.registry = registry
            self.pipeline.reset(registry)
            self.enrollment, self.pending_profile, self.mode = None, None, 'preview'
            self.emit('registry', names=registry.names(), warnings=registry.warnings(self.models.embedder_hash))
            self.emit('mode', mode=self.mode)

    def enroll_frame(self, frame):
        e = self.enrollment
        e.frames += 1
        boxes = self.models.detect(frame.image)
        output = [Observation(-1, b.tolist(), float(c), quality='enrollment') for b,c in zip(boxes.xyxy, boxes.conf)]
        usable = [(b, float(c)) for b,c in zip(boxes.xyxy, boxes.conf) if c >= self.settings.track_high]
        message = 'Show exactly ONE marker to enroll; other markers must be outside the frame.'
        # Weak second detections can still belong to another physical tag. Pause
        # collection rather than mix that appearance into a permanent profile.
        plausible_count = int(np.count_nonzero(boxes.conf > self.settings.track_low))
        if len(usable) == 1 and plausible_count == 1:
            crop, quality, _ = crop_quality(frame.image, usable[0][0], self.settings)
            if crop is not None:
                if frame.timestamp-e.last_sample >= e.interval and len(e.groups[e.step]) < e.samples_per_view:
                    e.add(self.models.embed([crop])[0], frame.timestamp)
                message = 'View complete. Click Next view.' if len(e.groups[e.step]) >= e.samples_per_view else 'Collecting clear samples. Keep moving gently.'
            else:
                message = 'Move closer or hold steadier: ' + quality.replace('_',' ')
        if e.complete:
            message = 'All views collected. Click Save enrollment to review and save.'
        self.emit('enrollment', step=e.step, view=VIEWS[e.step], count=len(e.groups[e.step]),
                  required=e.samples_per_view, total=e.count, complete=e.complete, message=message)
        return output

    def run(self):
        try:
            self.models = Models(self.settings)
            try:
                self.registry = Registry.load(self.settings.path('registry')) if self.settings.path('registry').exists() else Registry()
            except (ValueError, OSError) as exc:
                self.registry = Registry()
                self.emit('error', message=f'Registry could not be loaded: {exc}. Original file preserved; import a valid registry before saving.')
            self.pipeline = Pipeline(self.models, self.registry, self.settings)
            self.emit('ready', device=self.models.device)
            self.emit('registry', names=self.registry.names(), warnings=self.registry.warnings(self.models.embedder_hash))
            while not self.stopping.is_set():
                try:
                    command, args = self.commands.get(timeout=.01 if self.source else .1)
                    self.handle(command, args)
                except queue.Empty:
                    pass
                except Exception as exc:
                    self.emit('error', message=str(exc))
                if not self.source:
                    continue
                if (self.mode == 'enroll' and self.source_mode == 'video' and self.enrollment
                        and len(self.enrollment.groups[self.enrollment.step]) >= self.enrollment.samples_per_view):
                    # Keep a recording at the current view until the user advances.
                    continue
                try:
                    frame = self.source.read()
                    if self.source.error:
                        raise RuntimeError(self.source.error)
                    if frame is None:
                        if getattr(self.source,'ended',False):
                            self.stop_session()
                            self.source.close()
                            self.source = None
                            self.emit('status', message='Video ended. Reopen it to replay from the beginning.')
                        continue
                    observations, metrics = [], {}
                    if self.mode == 'run':
                        observations, metrics = self.pipeline.process(frame.image, frame.timestamp)
                        self.session.write(frame, observations, metrics, self.source.dropped)
                    elif self.mode == 'enroll':
                        observations = self.enroll_frame(frame)
                    item = (frame, observations, metrics, self.source.dropped)
                    try:
                        self.frames.put_nowait(item)
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        self.frames.put_nowait(item)
                except Exception as exc:
                    if self.session:
                        self.session.event('error', message=str(exc))
                    self.stop_session()
                    self.emit('error', message=str(exc))
                    self.mode = 'preview'
                    if self.source:
                        self.source.close()
                        self.source = None
        except Exception:
            self.emit('fatal', message=traceback.format_exc())
        finally:
            self.stop_session()
            if self.source:
                try:
                    self.source.close()
                except Exception as exc:
                    self.emit('error', message=str(exc))
