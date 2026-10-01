"""Detection, ByteTrack, ROI retention, quality checks, and identity resolution."""
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
import time
import numpy as np
from .identity import IdentityManager
from .vision import crop_quality, overlap


@dataclass
class Observation:
    track_id: int
    box: list
    confidence: float
    detected: bool = True
    name: str = 'unknown'
    score: float = -1.
    margin: float = 0.
    quality: str = 'pending'
    identity_status: str = 'pending'
    best_candidate: str = ''
    runner_up: str = ''
    threshold: float = 0.
    scores: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Pipeline:
    def __init__(self, models, registry, settings, tracker=None):
        self.models, self.settings = models, settings
        self.identities = IdentityManager(registry, settings, models.embedder_hash)
        if tracker is None:
            from ultralytics.trackers.byte_tracker import BYTETracker
            tracker = BYTETracker(SimpleNamespace(track_high_thresh=settings.track_high,
                                  track_low_thresh=settings.track_low, new_track_thresh=settings.new_track,
                                  match_thresh=settings.match_threshold, track_buffer=30, fuse_score=True))
        self.tracker = tracker
        self.rois = {}
        self.last_time = None

    def reset(self, registry=None):
        self.tracker.reset()
        self.rois.clear()
        self.identities = IdentityManager(registry or self.identities.matcher.registry,
                                          self.settings, self.models.embedder_hash)
        self.last_time = None

    def process(self, frame, now=None):
        now = time.monotonic() if now is None else now
        start = time.perf_counter()
        if self.last_time is not None and (now < self.last_time or now-self.last_time > self.settings.track_expiry_seconds):
            self.reset()
        self.last_time = now
        boxes = self.models.detect(frame)
        detection_done = time.perf_counter()
        # Bound tracker memory by elapsed time as well as ByteTrack's frame buffer.
        expired = {tid for tid, item in self.rois.items() if now-item['time'] > self.settings.track_expiry_seconds}
        for attr in ('tracked_stracks', 'lost_stracks'):
            if hasattr(self.tracker, attr):
                setattr(self.tracker, attr, [t for t in getattr(self.tracker, attr) if t.track_id not in expired])
        tracked = self.tracker.update(boxes, frame)
        tracking_done = time.perf_counter()
        output, crops, crop_ids, features = [], [], [], {}
        raw_boxes = np.asarray(boxes.xyxy)
        active = set()
        assigned_raw = set()
        for row in tracked:
            box, tid, confidence = np.asarray(row[:4], dtype=float), int(row[4]), float(row[5])
            # Ultralytics 8.4.10 returns the matched detection index within the
            # high- or low-confidence subset in its final column. Use that exact
            # association: rematching smoothed boxes can swap crops at crossings.
            if confidence >= self.settings.track_high:
                pool = np.flatnonzero(boxes.conf >= self.settings.track_high)
            else:
                pool = np.flatnonzero((boxes.conf > self.settings.track_low) &
                                      (boxes.conf < self.settings.track_high))
            local_index = int(row[7]) if len(row) == 8 else -1
            if not 0 <= local_index < len(pool):
                raise RuntimeError('Unexpected ByteTrack output; use the pinned Ultralytics dependency')
            owner = int(pool[local_index])
            if owner in assigned_raw or not np.isclose(boxes.conf[owner], confidence):
                raise RuntimeError('ByteTrack detection association is inconsistent')
            assigned_raw.add(owner)
            measured_box = raw_boxes[owner]
            active.add(tid)
            previous = self.rois.get(tid)
            velocity = np.zeros(4)
            if previous and now > previous['time']:
                velocity = (box-previous['box']) / (now-previous['time'])
                # Prevent a single association jump from projecting an ROI far away.
                max_speed = max(box[2]-box[0], box[3]-box[1]) * 3.
                velocity = np.clip(velocity, -max_speed, max_speed)
            self.rois[tid] = {'box':box, 'velocity':velocity, 'time':now, 'confidence':confidence}
            # Features come from the measured detection, not a lagging Kalman box.
            crop, quality, _ = crop_quality(frame, measured_box, self.settings)
            # A crop containing much of another marker is unsafe for updating an ID.
            if any(i != owner and overlap(measured_box, other) > self.settings.overlap_threshold
                   for i,other in enumerate(raw_boxes)):
                crop, quality = None, 'overlapping'
            output.append(Observation(tid, box.tolist(), confidence, quality=quality))
            features[tid] = None
            if crop is not None:
                crops.append(crop)
                crop_ids.append(tid)
        z = self.models.embed(crops)
        for tid, embedding in zip(crop_ids, z):
            features[tid] = embedding
        embedding_done = time.perf_counter()
        identities = self.identities.update(features, now)
        h, w = frame.shape[:2]
        for tid in list(self.rois):
            item = self.rois[tid]
            age = now-item['time']
            if age > self.settings.track_expiry_seconds:
                del self.rois[tid]
                continue
            if tid not in active and age <= self.settings.roi_hold_seconds:
                b = item['box'] + item['velocity'] * age
                b[[0,2]] = np.clip(b[[0,2]], 0, w-1)
                b[[1,3]] = np.clip(b[[1,3]], 0, h-1)
                if b[2] > b[0] and b[3] > b[1] and not any(overlap(b, o.box) > .6 for o in output):
                    output.append(Observation(tid, b.tolist(), item['confidence'], detected=False, quality='predicted_roi'))
        for o in output:
            state = identities.get(o.track_id)
            if state:
                o.name, o.score, o.margin, o.identity_status = state.name, state.score, state.margin, state.reason
                o.best_candidate, o.runner_up, o.threshold = state.best_name, state.runner_up, state.threshold
                o.scores = state.scores if o.detected else {}
        # Unconfirmed detections remain visible without pretending they are tracks.
        for i,(b, confidence) in enumerate(zip(raw_boxes, boxes.conf)):
            if i not in assigned_raw:
                output.append(Observation(-1, b.tolist(), float(confidence), quality='unconfirmed', identity_status='pending_track'))
        end = time.perf_counter()
        metrics = {'raw_detections':len(raw_boxes), 'tracked_detections':len(active),
                   'usable_crops':len(crops), 'predicted_rois':sum(not o.detected for o in output),
                   'identified':sum(o.detected and o.name != 'unknown' for o in output),
                   'unknown':sum(o.detected and o.name == 'unknown' for o in output),
                   'detector_ms':(detection_done-start)*1000, 'tracker_ms':(tracking_done-detection_done)*1000,
                   'embedding_ms':(embedding_done-tracking_done)*1000, 'identity_ms':(end-embedding_done)*1000,
                   'processing_ms':(end-start)*1000}
        metrics['raw_boxes'] = [{'box':b.tolist(), 'confidence':float(c)} for b,c in zip(raw_boxes,boxes.conf)]
        return output, metrics
