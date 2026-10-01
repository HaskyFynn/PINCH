"""Incremental logs with model provenance and actual elapsed runtime."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import time
import uuid
import random
import numpy as np


class Session:
    def __init__(self, settings, models, registry, source, record_video=False):
        self.directory = settings.path('runs') / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True)
        self.started = time.monotonic()
        self.count = 0
        self.latencies = []
        self.random = random.Random(0)
        self.record_video = record_video
        self.video = None
        self.video_times = None
        self.closed = False
        self.file = (self.directory / 'frames.jsonl').open('w', encoding='utf-8')
        self.events = (self.directory / 'events.jsonl').open('w', encoding='utf-8')
        versions = {name:importlib.metadata.version(name) for name in ('numpy','torch','torchvision','ultralytics','opencv-python')}
        metadata = {'schema_version':1, 'started_utc':datetime.now(timezone.utc).isoformat(),
                    'settings':settings.to_dict(), 'source':source, 'versions':versions,
                    'detector_sha256':models.detector_hash, 'embedder_sha256':models.embedder_hash,
                    'registry':registry.to_json(), 'device':models.device,
                    'recording':'processed raw frames with original timestamps' if record_video else 'disabled',
                    'ground_truth':'not supplied; no recognition accuracy is inferred from predictions'}
        (self.directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def write(self, frame, observations, metrics, dropped=0):
        if self.record_video:
            import cv2
            if self.video is None:
                h,w=frame.image.shape[:2]
                self.video=cv2.VideoWriter(str(self.directory/'capture.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10.,(w,h))
                if not self.video.isOpened():
                    raise RuntimeError('Cannot create session recording; check storage and video codec support')
                self.video_times=(self.directory/'video_timestamps.jsonl').open('w',encoding='utf-8')
            self.video.write(frame.image)
            self.video_times.write(json.dumps({'video_frame':self.count,'source_frame':frame.index,'timestamp':frame.timestamp})+'\n')
            self.video_times.flush()
        payload = {'frame':frame.index, 'timestamp':frame.timestamp,
                   'recorded_frame':self.count if self.record_video else None,
                   'source_read_to_result_ms':(time.monotonic()-frame.captured_at)*1000,
                   'capture_frames_skipped':dropped, 'metrics':metrics,
                   'observations':[o.to_dict() for o in observations]}
        self.file.write(json.dumps(payload, allow_nan=False) + '\n')
        self.file.flush()
        self.count += 1
        # Bounded uniform reservoir for long camera sessions.
        if len(self.latencies)<20000:
            self.latencies.append(metrics['processing_ms'])
        else:
            i=self.random.randrange(self.count)
            if i<len(self.latencies):
                self.latencies[i]=metrics['processing_ms']

    def event(self, kind, **data):
        self.events.write(json.dumps({'event':kind, 'elapsed_seconds':time.monotonic()-self.started, **data}) + '\n')
        self.events.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        elapsed = time.monotonic()-self.started
        self.file.close()
        self.events.close()
        if self.video is not None:
            self.video.release()
        if self.video_times is not None:
            self.video_times.close()
        summary = {'processed_frames':self.count, 'elapsed_seconds':elapsed,
                   'processing_fps':self.count/max(elapsed, 1e-6),
                   'median_processing_ms':float(np.median(self.latencies)) if self.latencies else None,
                   'p95_processing_ms':float(np.percentile(self.latencies,95)) if self.latencies else None,
                   'latency_sample_count':len(self.latencies),
                   'identity_accuracy':None}
        (self.directory / 'summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
