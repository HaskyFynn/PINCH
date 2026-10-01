"""Latest-frame camera capture; deterministic sequential video replay."""
from dataclasses import dataclass
import threading
import time
import cv2
import json
import math
from pathlib import Path


@dataclass
class Frame:
    image: object
    index: int
    timestamp: float
    captured_at: float


class CameraSource:
    def __init__(self, index, settings):
        self.index, self.settings = index, settings
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.consumed = -1
        self.error = ''
        self.dropped = 0
        self.thread = threading.Thread(target=self._capture, name='pinch-camera', daemon=True)
        self.thread.start()

    def _capture(self):
        cap = cv2.VideoCapture(self.index)
        try:
            if not cap.isOpened():
                self.error = f'Cannot open camera {self.index}. Close other camera apps or choose another index.'
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            i, failures = 0, 0
            while not self.stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        self.error = 'Camera stopped delivering frames. Reopen the camera to retry.'
                        return
                    self.stop_event.wait(.05)
                    continue
                failures = 0
                now = time.monotonic()
                with self.lock:
                    if self.latest is not None and self.latest.index != self.consumed:
                        self.dropped += 1
                    self.latest = Frame(image, i, now, now)
                i += 1
        finally:
            cap.release()

    def read(self):
        with self.lock:
            if self.latest is None or self.latest.index == self.consumed:
                return None
            self.consumed = self.latest.index
            return self.latest

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('Camera driver has not released the device yet; close PINCH before retrying.')


class VideoSource:
    def __init__(self, path, paced=True):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f'Cannot open video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.
        self.index = 0
        self.paced = paced
        self.next_frame_at = time.monotonic()
        self.timeline = None
        timeline = Path(path).with_name('video_timestamps.jsonl')
        if Path(path).name=='capture.avi' and timeline.exists():
            try:
                entries=[json.loads(line) for line in timeline.read_text(encoding='utf-8').splitlines()]
                if any(item['video_frame']!=i for i,item in enumerate(entries)):
                    raise ValueError('Recording timestamps are incomplete or out of order')
                self.timeline=[float(item['timestamp']) for item in entries]
                if (any(not math.isfinite(t) for t in self.timeline) or
                        any(b <= a for a,b in zip(self.timeline,self.timeline[1:]))):
                    raise ValueError('Recording timestamps must be finite and strictly increasing')
            except Exception:
                self.cap.release()
                raise
        self.error = ''
        self.ended = False
        self.dropped = 0

    def read(self):
        if self.paced and time.monotonic() < self.next_frame_at:
            return None
        ok, image = self.cap.read()
        if not ok:
            self.ended = True
            return None
        if self.timeline is not None and self.index>=len(self.timeline):
            self.error='Recording is missing original frame timestamps'
            self.ended=True
            return None
        timestamp=self.timeline[self.index] if self.timeline is not None else self.index/self.fps
        result = Frame(image, self.index, timestamp, time.monotonic())
        self.index += 1
        self.next_frame_at = time.monotonic() + 1 / self.fps
        return result

    def close(self):
        self.cap.release()
