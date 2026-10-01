"""Guided enrollment accepts one physical marker at a time."""
from dataclasses import dataclass, field
import numpy as np
from .registry import DIM, MarkerProfile, unit_rows

VIEWS = ('Front, move gently', 'Tilt left and right', 'Move nearer and farther',
         'Move across the frame', 'Rotate and tilt gently')


def prototypes(x, k=5):
    x = unit_rows(x)
    # Deterministic farthest-point seeds give varied views a chance to contribute.
    centers = [x[0]]
    for _ in range(1, min(k, len(x))):
        centers.append(x[np.argmin(np.max(x @ np.asarray(centers).T, axis=1))])
    c = np.asarray(centers)
    for _ in range(18):
        labels = np.argmax(x @ c.T, axis=1)
        for i in range(len(c)):
            members = x[labels == i]
            if len(members):
                mean = members.mean(axis=0)
                c[i] = mean / max(np.linalg.norm(mean), 1e-8)
    return c


@dataclass
class Enrollment:
    name: str
    samples_per_view: int = 8
    interval: float = .15
    groups: list = field(default_factory=lambda: [[] for _ in VIEWS])
    step: int = 0
    last_sample: float = -float('inf')
    frames: int = 0

    @property
    def count(self):
        return sum(map(len, self.groups))

    @property
    def complete(self):
        return all(len(g) >= self.samples_per_view for g in self.groups)

    def add(self, embedding, now):
        if now - self.last_sample < self.interval or self.complete:
            return False
        vector = unit_rows(np.asarray(embedding).reshape(1, -1))[0]
        if len(self.groups[self.step]) >= self.samples_per_view:
            return False
        self.groups[self.step].append(vector)
        self.last_sample = now
        return True

    def advance(self):
        if len(self.groups[self.step]) < self.samples_per_view:
            raise ValueError('Collect enough clear samples for this view first')
        if self.step < len(VIEWS) - 1:
            self.step += 1

    def build(self, model_hash='', source_mode='camera', source_path=''):
        if not self.complete:
            raise ValueError('Finish each view before saving; the previous profile is unchanged')
        # Reserve the final temporal block in EVERY view. Do not fit prototypes to it.
        train, validation = [], []
        for group in self.groups:
            hold = max(2, len(group) // 4)
            train.extend(group[:-hold])
            validation.extend(group[-hold:])
        p = prototypes(train)
        validation = unit_rows(validation)
        scores = np.clip((validation @ p.T).max(axis=1), -1., 1.)
        threshold = float(np.percentile(scores, 5))
        all_samples = unit_rows([z for group in self.groups for z in group])
        return MarkerProfile(self.name, p.tolist(), threshold,
                             enroll_frames=self.frames, enroll_used=len(all_samples),
                             enroll_embs=all_samples.tolist(), embedder_sha256=model_hash,
                             calibration='within_session_temporal_holdout',
                             source_mode=source_mode, source_path=source_path).validate()
