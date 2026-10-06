"""Explicit, validated configuration. Relative paths belong to the project."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    detector: str = "letsgo/best.pt"
    embedder: str = "letsgo/embedder_resnet18_triplet.pt"
    registry: str = "data/registry.json"
    runs: str = "runs/live"
    device: str = "cpu"
    image_size: int = 640
    detector_confidence: float = 0.08
    detector_iou: float = 0.7
    max_detections: int = 32
    track_high: float = 0.25
    track_low: float = 0.1
    new_track: float = 0.25
    match_threshold: float = 0.8
    min_crop_area: int = 1600
    blur_threshold: float = 25.0
    crop_padding: float = 0.06
    overlap_threshold: float = 0.45
    identity_margin: float = 0.01
    identity_confirm_frames: int = 3
    identity_confirmation_gap: float = 3.0
    identity_hold_seconds: float = 0.6
    roi_hold_seconds: float = 0.4
    track_expiry_seconds: float = 3.0
    enrollment_samples_per_view: int = 8
    enrollment_interval: float = 0.15
    cpu_threads: int = 4
    camera_width: int = 1280
    camera_height: int = 720
    use_density: bool = False

    def path(self, field):
        p = Path(getattr(self, field)).expanduser()
        return p if p.is_absolute() else ROOT / p

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            default = f.default
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{f.name} must be true or false")
            elif isinstance(default, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{f.name} must be a finite number")
                if isinstance(default, int) and not isinstance(value, int):
                    raise ValueError(f"{f.name} must be an integer")
                if value <= 0:
                    raise ValueError(f"{f.name} must be positive")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{f.name} must be a nonempty string")
        for key in ('detector_confidence', 'detector_iou', 'track_high', 'track_low',
                    'new_track', 'match_threshold', 'overlap_threshold', 'identity_margin'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f"{key} must be between zero and one")
        if not self.detector_confidence <= self.track_low < self.track_high <= self.new_track:
            raise ValueError("Require detector_confidence <= track_low < track_high <= new_track")
        if not self.roi_hold_seconds <= self.identity_hold_seconds < self.track_expiry_seconds:
            raise ValueError("Require ROI hold <= identity hold < track expiry")
        if self.max_detections < 4 or self.enrollment_samples_per_view < 6:
            raise ValueError("Allow at least four detections and six enrollment samples per view")
        return self

    def to_dict(self):
        return asdict(self)


def load_settings(path=None):
    path = Path(path) if path else ROOT / 'config' / 'live.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    unknown = set(data) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return Settings(**data).validate()
