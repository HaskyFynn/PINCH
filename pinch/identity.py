"""Cached prototype scoring and bounded, exclusive temporal identities."""
from dataclasses import dataclass, field
import numpy as np
from .registry import DIM, unit_rows


class Matcher:
    def __init__(self, registry, model_hash='', use_density=False):
        self.registry = registry
        self.names = registry.names()
        self.use_density = use_density
        self.thresholds = np.array([m.thr for m in registry.markers], dtype=np.float64)
        self.slices = []
        arrays = []
        start = 0
        for m in registry.markers:
            a = unit_rows(m.proto)
            arrays.append(a)
            self.slices.append(slice(start, start + len(a)))
            start += len(a)
        self.matrix = np.concatenate(arrays) if arrays else np.empty((0, DIM), dtype=np.float32)
        self.compatible = np.array([not m.embedder_sha256 or not model_hash or
                                    m.embedder_sha256 == model_hash for m in registry.markers])

    def score(self, embeddings):
        z = unit_rows(embeddings)
        scores = np.empty((len(z), len(self.names)), dtype=np.float32)
        dots = z @ self.matrix.T
        for i, sl in enumerate(self.slices):
            scores[:, i] = np.max(dots[:, sl], axis=1)
        return np.clip(scores, -1., 1.)

    def candidate(self, embedding, margin, scores=None):
        if scores is None:
            scores = self.score(np.asarray(embedding).reshape(1, -1))[0]
        if not len(scores):
            return None, -1., 0., 'no_profiles'
        order = np.argsort(-scores)
        k = int(order[0])
        best = float(scores[k])
        gap = best - float(scores[order[1]]) if len(order) > 1 else 2.
        if not self.compatible[k]:
            return None, best, gap, 'model_mismatch'
        if best < self.thresholds[k]:
            return None, best, gap, 'below_threshold'
        if gap < margin:
            return None, best, gap, 'ambiguous'
        m = self.registry.markers[k]
        # Older profiles intentionally lack these optional statistics.
        if self.use_density and m.mean is not None and m.var is not None and m.ll_thr is not None:
            z = unit_rows(np.asarray(embedding).reshape(1, -1))[0]
            ll = float(-.5 * np.sum((z - np.asarray(m.mean)) ** 2 / np.asarray(m.var)))
            if ll < m.ll_thr:
                return None, best, gap, 'density_rejected'
        return self.names[k], best, gap, 'accepted'


@dataclass
class IdentityState:
    name: str = 'unknown'
    score: float = -1.
    margin: float = 0.
    pending: str | None = None
    count: int = 0
    last_evidence: float = -float('inf')
    last_update: float = -float('inf')
    last_seen: float = 0.
    reason: str = 'pending'
    switches: int = 0
    best_name: str = ''
    runner_up: str = ''
    threshold: float = 0.
    scores: dict = field(default_factory=dict)


class IdentityManager:
    def __init__(self, registry, settings, model_hash=''):
        self.settings = settings
        self.matcher = Matcher(registry, model_hash, settings.use_density)
        self.states = {}

    def update(self, observations, now):
        """observations maps track ID to an embedding, or None for an unusable crop.

        Each track must independently pass absolute and runner-up gates. Joint
        conflict resolution never assigns a losing track its weaker second choice.
        """
        proposals = {}
        valid_ids = [tid for tid,z in observations.items() if z is not None]
        score_rows = self.matcher.score([observations[tid] for tid in valid_ids]) if valid_ids else []
        scores_by_id = dict(zip(valid_ids,score_rows))
        for tid, z in observations.items():
            state = self.states.setdefault(tid, IdentityState())
            state.last_seen = now
            if now - state.last_update > self.settings.identity_confirmation_gap:
                state.count, state.pending = 0, None
            if z is not None:
                scores = scores_by_id[tid]
                proposals[tid] = self.matcher.candidate(z, self.settings.identity_margin, scores)
                state.scores = dict(zip(self.matcher.names,map(float,scores)))
                if len(scores):
                    order = np.argsort(-scores)
                    state.best_name = self.matcher.names[order[0]]
                    state.runner_up = self.matcher.names[order[1]] if len(order)>1 else ''
                    state.threshold = float(self.matcher.thresholds[order[0]])
            else:
                state.scores = {}
                proposals[tid] = (None, state.score, 0., 'quality_hold')

        # One physical marker per registered name. Similar-strength conflicts are
        # left unknown; a clear winner can reacquire after a tracker ID changes.
        for name in self.matcher.names:
            contenders = [tid for tid, p in proposals.items() if p[0] == name]
            if len(contenders) <= 1:
                continue
            contenders.sort(key=lambda tid: proposals[tid][1], reverse=True)
            first, second = contenders[:2]
            if proposals[first][1] - proposals[second][1] < self.settings.identity_margin:
                losers = contenders
            else:
                losers = contenders[1:]
            for tid in losers:
                _, score, gap, _ = proposals[tid]
                proposals[tid] = (None, score, gap, 'identity_conflict')

        for tid, (name, score, gap, reason) in proposals.items():
            state = self.states[tid]
            previous = state.name
            state.score, state.margin, state.reason = score, gap, reason
            if name is not None:
                # Strong contradictory evidence must not preserve the old name.
                if state.name not in ('unknown', name):
                    state.name = 'unknown'
                    state.switches += 1
                state.count = state.count + 1 if state.pending == name else 1
                state.pending = name
                state.last_update = now
                if state.count >= self.settings.identity_confirm_frames or state.name == name:
                    state.name = name
                    state.last_evidence = now
                    state.reason = 'identified'
                else:
                    state.reason = 'confirming'
            else:
                state.pending, state.count = None, 0
                if reason in ('identity_conflict', 'model_mismatch'):
                    state.name = 'unknown'
                elif now - state.last_evidence > self.settings.identity_hold_seconds:
                    state.name = 'unknown'
                elif state.name != 'unknown':
                    state.reason = 'held_' + reason
            if previous != 'unknown' and state.name == 'unknown' and name is None:
                state.switches += 1

        # Expire on elapsed time even on frames with no detections or embeddings.
        for tid in list(self.states):
            state = self.states[tid]
            if now - state.last_seen > self.settings.track_expiry_seconds:
                del self.states[tid]
                continue
            if now - state.last_evidence > self.settings.identity_hold_seconds:
                state.name = 'unknown'
            if tid not in observations:
                state.pending, state.count = None, 0
                state.reason = 'temporarily_lost'

        # A newly confirmed owner supersedes a name retained by an absent/blurred
        # track. Equal fresh evidence is already rejected above.
        owners = {}
        for tid, state in self.states.items():
            if state.name != 'unknown':
                owners.setdefault(state.name, []).append(tid)
        for tids in owners.values():
            if len(tids) > 1:
                tids.sort(key=lambda t: (self.states[t].last_evidence, self.states[t].score), reverse=True)
                for tid in tids[1:]:
                    self.states[tid].name = 'unknown'
                    self.states[tid].reason = 'identity_conflict'
        return self.states
