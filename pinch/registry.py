"""Versioned marker profiles, validation, and atomic persistence."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import os
import shutil
import tempfile
import numpy as np

DIM = 128


def unit_rows(values, name='embeddings'):
    try:
        a = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: invalid numeric array') from exc
    if a.ndim != 2 or a.shape[1] != DIM or len(a) == 0:
        raise ValueError(f'{name}: expected one or more {DIM}-dimensional vectors')
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(a).all() or (norms < 1e-8).any():
        raise ValueError(f'{name}: vectors must be finite and nonzero')
    return a / norms


@dataclass
class MarkerProfile:
    marker_id: str
    proto: list
    thr: float
    enroll_frames: int = 0
    enroll_used: int = 0
    source_mode: str = ''
    source_path: str = ''
    mean: list | None = None
    var: list | None = None
    ll_thr: float | None = None
    enroll_embs: list | None = None
    embedder_sha256: str = ''
    calibration: str = 'legacy'

    def validate(self):
        if not isinstance(self.marker_id, str) or not self.marker_id.strip() or self.marker_id.strip().lower() == 'unknown' or len(self.marker_id.strip())>64:
            raise ValueError('Marker name must be nonempty and cannot be "unknown"')
        self.marker_id = self.marker_id.strip()
        self.proto = unit_rows(self.proto, self.marker_id + ' prototypes').tolist()
        if not isinstance(self.thr, (int, float)) or not np.isfinite(self.thr) or not -1 <= self.thr <= 1:
            raise ValueError(f'{self.marker_id}: invalid cosine threshold')
        if self.enroll_embs is not None:
            self.enroll_embs = unit_rows(self.enroll_embs, self.marker_id + ' enrollment').tolist()
        if self.mean is not None or self.var is not None:
            try:
                m = np.asarray(self.mean, dtype=np.float64)
                v = np.asarray(self.var, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{self.marker_id}: invalid density statistics') from exc
            if m.shape != (DIM,) or v.shape != (DIM,) or not np.isfinite(m).all() or not np.isfinite(v).all() or (v <= 0).any():
                raise ValueError(f'{self.marker_id}: invalid density statistics')
            if not isinstance(self.ll_thr, (int, float)) or not np.isfinite(self.ll_thr):
                raise ValueError(f'{self.marker_id}: missing density threshold')
        return self


class Registry:
    def __init__(self, markers=()):
        self.markers = []
        for m in markers:
            m.validate()
            if m.marker_id in self.names():
                raise ValueError(f'Duplicate marker name: {m.marker_id}')
            self.markers.append(m)

    def names(self):
        return [m.marker_id for m in self.markers]

    def with_profile(self, profile):
        return Registry([m for m in self.markers if m.marker_id != profile.marker_id] + [profile])

    def warnings(self, embedder_hash=''):
        result = []
        for m in self.markers:
            if not m.embedder_sha256:
                result.append(f'{m.marker_id}: legacy profile; verify with the current model or re-enroll.')
            elif embedder_hash and m.embedder_sha256 != embedder_hash:
                result.append(f'{m.marker_id}: different embedding model; re-enrollment required.')
        return result

    def to_json(self):
        return {'schema_version': 2, 'markers': [asdict(m) for m in self.markers]}

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict) or data.get('schema_version', 1) not in (1, 2):
            raise ValueError('Unsupported registry format')
        markers = data.get('markers')
        if not isinstance(markers, list):
            raise ValueError('Registry must contain a markers list')
        allowed = {f.name for f in fields(MarkerProfile)}
        profiles = []
        for item in markers:
            if not isinstance(item, dict) or set(item) - allowed:
                raise ValueError('Invalid marker fields; original registry was not changed')
            try:
                profiles.append(MarkerProfile(**item))
            except TypeError as exc:
                raise ValueError('Marker requires marker_id, proto, and thr') from exc
        return cls(profiles)

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def save(self, path):
        # Validate the complete replacement before touching the last good file.
        payload = Registry.from_json(self.to_json()).to_json()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            if path.exists():
                shutil.copy2(path, str(path) + '.bak')
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
