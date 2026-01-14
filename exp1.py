#!/usr/bin/env python3
"""
PINCH Robustness Runner (UI + full marker pipeline)
- Enrollment: multi-prototype registration (covers marker appearance modes under motion/tilt)
- Runtime ID: max-sim over prototypes + margin rule + temporal stability (N consecutive frames)
- Robustness trials: labels for lighting + distance, auto logging to CSV + JSON summaries
- Real-time: optional frame buffer draining to avoid webcam backlog (keeps latency honest)
- Detector backends:
  A) TFLite YOLO (default): uses a .tflite beside this script or in ./assets
  B) Ultralytics YOLO (.pt) with ByteTrack (optional): requires ultralytics installed

Folder layout (same parent folder as this script):
- Put your model(s) here or in ./assets:
  - TFLite: *.tflite
  - PT: *.pt
- Enrollment DB: ./user_gestures/*.json
- Logs: ./logs/session_YYYYMMDD_HHMMSS/

Participant workflow (no keyboard needed):
1) Enroll each participant one-by-one (only one marker visible): click Enroll
2) Set roster left-to-right (comma list) once: click Reload DB then Set Roster
3) For each lighting/distance condition: choose dropdowns, click Start Trial
4) Trial auto stops after duration, logs files, then repeat next condition

Important:
- Robustness trials assume each participant stays roughly in their left-to-right slot.
- Crossing/occlusion interference is a separate experiment. This script focuses on robustness.

"""

import argparse
import json
import math
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

# Tkinter UI (built-in)
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------
# Defaults
# ---------------------------

DEFAULT_LIGHTS = ["Bright", "Dim", "Daylight"]
DEFAULT_DISTANCES = ["Near", "Mid", "Far"]

REG_SECONDS_DEFAULT = 8.0
TRIAL_SECONDS_DEFAULT = 10.0

MIN_CONF_DEFAULT = 0.45
SIM_THRESHOLD_DEFAULT = 0.75
MARGIN_MIN_DEFAULT = 0.05

PROTO_K_DEFAULT = 8
PROTO_DUP_SIM_DEFAULT = 0.93

STABLE_N_DEFAULT = 2

TRACK_ASSIGN_DIST_PX = 120.0
TRACK_TTL_SEC = 1.0

MAX_DRAW_TRACKS = 12


# ---------------------------
# Utils
# ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def auto_find_model(root: Path, exts: Tuple[str, ...]) -> Optional[str]:
    candidates: List[Path] = []
    for p in [root, root / "assets"]:
        if p.exists():
            for e in exts:
                candidates += list(p.glob(f"*{e}"))
    candidates = sorted(candidates)
    return str(candidates[0]) if candidates else None

def write_csv_header(fp, cols: List[str]):
    fp.write(",".join(cols) + "\n")

def write_csv_row(fp, cols: List[str], row: Dict):
    vals = []
    for c in cols:
        v = row.get(c, "")
        if isinstance(v, float):
            vals.append(f"{v:.6f}")
        else:
            vals.append(str(v))
    fp.write(",".join(vals) + "\n")

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


# ---------------------------
# Detector backends
# ---------------------------

@dataclass
class DetOut:
    box: List[float]          # [x1,y1,x2,y2]
    score: float
    det_id: Optional[int] = None  # if backend gives a track id

class TFLiteYoloDetector:
    """
    YOLO-like TFLite output: [1,5,N] with channels cx,cy,w,h,obj.
    """

    def __init__(self, model_path: str, num_threads: int = 2, min_conf: float = 0.45):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        in_shape = self.input_details[0]["shape"]
        if len(in_shape) != 4:
            raise ValueError(f"Expected input [1,H,W,C], got {in_shape}")
        self.in_size = int(in_shape[1])

        out_shape = self.output_details[0]["shape"]
        if not (len(out_shape) == 3 and int(out_shape[1]) == 5):
            raise ValueError(f"Expected output [1,5,N], got {out_shape}")
        self.out_count = int(out_shape[2])

        self.min_conf = float(min_conf)

        self.n80 = 80 * 80
        self.n40 = 40 * 40

    @staticmethod
    def _nms(boxes, scores, iou_thresh=0.45):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def _decode_box(self, idx: int, cx_arr, cy_arr, w_arr, h_arr):
        n80 = self.n80
        n40 = self.n40

        if idx < n80:
            g, offset = 80, 0
        elif idx < n80 + n40:
            g, offset = 40, n80
        else:
            g, offset = 20, n80 + n40

        cell = idx - offset
        gy = cell // g
        gx = cell % g

        grid_cx = (gx + 0.5) / float(g)
        grid_cy = (gy + 0.5) / float(g)

        net_cx = float(np.clip(cx_arr[idx], 0.0, 1.0))
        net_cy = float(np.clip(cy_arr[idx], 0.0, 1.0))

        cx = float(np.clip(0.4 * net_cx + 0.6 * grid_cx, 0.0, 1.0))
        cy = float(np.clip(0.4 * net_cy + 0.6 * grid_cy, 0.0, 1.0))

        w = float(np.clip(w_arr[idx], 0.05, 1.0))
        h = float(np.clip(h_arr[idx], 0.05, 1.0))

        cx_px = cx * self.in_size
        cy_px = cy * self.in_size
        w_px = w * self.in_size
        h_px = h * self.in_size

        left = float(np.clip(cx_px - w_px / 2.0, 0.0, self.in_size - 1.0))
        top = float(np.clip(cy_px - h_px / 2.0, 0.0, self.in_size - 1.0))
        right = float(np.clip(cx_px + w_px / 2.0, left + 1.0, float(self.in_size)))
        bottom = float(np.clip(cy_px + h_px / 2.0, top + 1.0, float(self.in_size)))
        return left, top, right, bottom

    def detect(self, frame_bgr) -> List[DetOut]:
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        inp = resized.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]  # [5,N]
        if out.shape[0] != 5:
            return []

        cx_arr, cy_arr, w_arr, h_arr, obj_arr = out

        boxes_model = []
        scores = []
        for i in range(self.out_count):
            s = float(obj_arr[i])
            if s >= self.min_conf:
                l, t, r, b = self._decode_box(i, cx_arr, cy_arr, w_arr, h_arr)
                boxes_model.append([l, t, r, b])
                scores.append(s)

        if not boxes_model:
            return []

        keep = self._nms(boxes_model, scores, iou_thresh=0.45)

        sx = w / float(self.in_size)
        sy = h / float(self.in_size)

        dets = []
        for i in keep:
            l, t, r, b = boxes_model[i]
            dets.append(DetOut([l * sx, t * sy, r * sx, b * sy], float(scores[i]), None))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets


class UltralyticsYoloTracker:
    """
    Optional backend: YOLO(.pt) + ByteTrack
    Requires: pip install ultralytics
    """
    def __init__(self, model_path: str, tracker_yaml: str = "bytetrack.yaml", conf: float = 0.45, iou: float = 0.5):
        from ultralytics import YOLO  # type: ignore
        self.model = YOLO(model_path)
        self.tracker_yaml = tracker_yaml
        self.conf = float(conf)
        self.iou = float(iou)

    def detect(self, frame_bgr) -> List[DetOut]:
        # persist=True keeps track IDs across frames
        r = self.model.track(
            frame_bgr,
            tracker=self.tracker_yaml,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else None
        ids = None
        try:
            ids = r.boxes.id.detach().cpu().numpy() if r.boxes.id is not None else None
        except Exception:
            ids = None

        dets: List[DetOut] = []
        for i in range(xyxy.shape[0]):
            box = xyxy[i].tolist()
            score = float(confs[i]) if confs is not None else 1.0
            det_id = int(ids[i]) if ids is not None else None
            dets.append(DetOut(box=box, score=score, det_id=det_id))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets


# ---------------------------
# Embedder (your pipeline kept)
# ---------------------------

class HybridPatchEmbedder:
    SIZE = 64
    PATCH_SIZE = 32
    CELLS = 4
    ANGLE_BINS = 8
    COLOR_BINS = 16

    PIX_DIM = PATCH_SIZE * PATCH_SIZE
    GRAD_DIM = CELLS * CELLS * ANGLE_BINS
    DIM = PIX_DIM + GRAD_DIM + COLOR_BINS * 3

    CENTER_CROP_RATIO = 0.90

    PIX_WEIGHT = 0.35
    GRAD_WEIGHT = 2.0
    COLOR_WEIGHT = 3.0

    def embed(self, frame_bgr, box_xyxy):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = int(max(0, min(w - 2, x1)))
        y1 = int(max(0, min(h - 2, y1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y2 = int(max(y1 + 1, min(h, y2)))

        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        half_w = bw * self.CENTER_CROP_RATIO * 0.5
        half_h = bh * self.CENTER_CROP_RATIO * 0.5

        l = int(round(cx - half_w))
        t = int(round(cy - half_h))
        r = int(round(cx + half_w))
        b = int(round(cy + half_h))

        l = max(0, min(w - 2, l))
        t = max(0, min(h - 2, t))
        r = max(l + 1, min(w, r))
        b = max(t + 1, min(h, b))

        crop = frame_bgr[t:b, l:r, :]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)

        b_ch, g_ch, r_ch = cv2.split(resized)
        gray = (0.114 * b_ch + 0.587 * g_ch + 0.299 * r_ch) / 255.0
        gray = gray.astype(np.float32)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.float32) * 2.0
        S = hsv[:, :, 1].astype(np.float32) / 255.0
        V = hsv[:, :, 2].astype(np.float32) / 255.0

        mean = float(np.mean(gray))
        var = float(np.mean(gray * gray) - mean * mean)
        std = math.sqrt(max(var, 1e-6))
        gray = (gray - mean) / std
        gray_flat = gray.flatten()

        feat_pix = np.zeros(self.PIX_DIM, dtype=np.float32)
        step = self.SIZE // self.PATCH_SIZE
        k = 0
        for yy in range(0, self.SIZE, step):
            for xx in range(0, self.SIZE, step):
                if k >= self.PIX_DIM:
                    break
                feat_pix[k] = gray_flat[yy * self.SIZE + xx]
                k += 1
            if k >= self.PIX_DIM:
                break

        feat_grad = np.zeros(self.GRAD_DIM, dtype=np.float32)
        cell_w = self.SIZE // self.CELLS
        cell_h = self.SIZE // self.CELLS

        for yy in range(1, self.SIZE - 1):
            for xx in range(1, self.SIZE - 1):
                idx0 = yy * self.SIZE + xx
                gx = gray_flat[idx0 + 1] - gray_flat[idx0 - 1]
                gy = gray_flat[idx0 + self.SIZE] - gray_flat[idx0 - self.SIZE]
                mag = math.sqrt(float(gx * gx + gy * gy))
                if mag <= 0.0:
                    continue
                angle = math.atan2(float(gy), float(gx))
                if angle < 0.0:
                    angle += math.pi
                bin_idx = int((angle / math.pi) * self.ANGLE_BINS)
                bin_idx = max(0, min(self.ANGLE_BINS - 1, bin_idx))

                cx_cell = min(self.CELLS - 1, xx // cell_w)
                cy_cell = min(self.CELLS - 1, yy // cell_h)
                cell_index = cy_cell * self.CELLS + cx_cell
                feat_index = cell_index * self.ANGLE_BINS + bin_idx
                feat_grad[feat_index] += mag

        histH = np.zeros(self.COLOR_BINS, dtype=np.float32)
        histS = np.zeros(self.COLOR_BINS, dtype=np.float32)
        histV = np.zeros(self.COLOR_BINS, dtype=np.float32)

        for yy in range(self.SIZE):
            for xx in range(self.SIZE):
                h_val = H[yy, xx]
                s_val = S[yy, xx]
                v_val = V[yy, xx]
                h_bin = int((h_val / 360.0) * self.COLOR_BINS)
                s_bin = int(s_val * self.COLOR_BINS)
                v_bin = int(v_val * self.COLOR_BINS)
                h_bin = max(0, min(self.COLOR_BINS - 1, h_bin))
                s_bin = max(0, min(self.COLOR_BINS - 1, s_bin))
                v_bin = max(0, min(self.COLOR_BINS - 1, v_bin))
                histH[h_bin] += 1.0
                histS[s_bin] += 1.0
                histV[v_bin] += 1.0

        def l1norm(x):
            s = float(np.sum(np.abs(x)))
            if s > 0:
                x /= s

        l1norm(histH)
        l1norm(histS)
        l1norm(histV)

        histH *= self.COLOR_WEIGHT
        histS *= self.COLOR_WEIGHT
        histV *= self.COLOR_WEIGHT

        feat_pix *= self.PIX_WEIGHT
        feat_grad *= self.GRAD_WEIGHT

        feat = np.zeros(self.DIM, dtype=np.float32)
        off = 0
        feat[off:off + self.PIX_DIM] = feat_pix
        off += self.PIX_DIM
        feat[off:off + self.GRAD_DIM] = feat_grad
        off += self.GRAD_DIM
        feat[off:off + self.COLOR_BINS] = histH
        off += self.COLOR_BINS
        feat[off:off + self.COLOR_BINS] = histS
        off += self.COLOR_BINS
        feat[off:off + self.COLOR_BINS] = histV

        nrm = math.sqrt(max(float(np.dot(feat, feat)), 1e-9))
        feat /= nrm
        return feat


# ---------------------------
# Prototypes + DB + matching
# ---------------------------

def normalize_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return (M / n).astype(np.float32)

def select_prototypes_greedy(embeds: List[np.ndarray], k: int, dup_sim: float) -> List[np.ndarray]:
    if not embeds:
        return []
    E = np.stack(embeds, axis=0).astype(np.float32)
    E = normalize_rows(E)

    m = np.mean(E, axis=0)
    m /= (np.linalg.norm(m) + 1e-9)

    sims_to_mean = E @ m
    first_idx = int(np.argmax(sims_to_mean))
    selected = [first_idx]

    best_sim = E @ E[first_idx]

    while len(selected) < k:
        cand_idx = int(np.argmin(best_sim))

        if float(best_sim[cand_idx]) >= dup_sim:
            ok = np.where(best_sim < dup_sim)[0]
            if ok.size == 0:
                break
            cand_idx = int(ok[np.argmin(best_sim[ok])])

        if cand_idx in selected:
            break

        selected.append(cand_idx)
        best_sim = np.maximum(best_sim, E @ E[cand_idx])

    return [E[i].copy() for i in selected]

class UserGestureDB:
    def __init__(self, root: Path):
        self.root = root
        ensure_dir(self.root)

    def _file_for_id(self, tag_id: str) -> Path:
        safe = tag_id.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.json"

    def save_user(self, tag_id: str, user_name: str, embeds: List[np.ndarray], k: int, dup_sim: float):
        if not embeds:
            return

        E = np.stack(embeds, axis=0).astype(np.float32)
        E = normalize_rows(E)

        mean_vec = np.mean(E, axis=0)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)

        protos = select_prototypes_greedy([e for e in E], k=k, dup_sim=dup_sim)
        if not protos:
            protos = [mean_vec.astype(np.float32)]

        js = {
            "tag_id": tag_id,
            "user_name": user_name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dim": int(E.shape[1]),
            "n_samples": int(E.shape[0]),
            "n_prototypes": int(len(protos)),
            "mean_vec": mean_vec.astype(np.float32).tolist(),
            "prototypes": [p.astype(np.float32).tolist() for p in protos],
        }

        path = self._file_for_id(tag_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(js, f)

    def load_all(self) -> List[dict]:
        users = []
        for p in sorted(self.root.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
                protos = np.array(js["prototypes"], dtype=np.float32)
                protos = normalize_rows(protos)
                users.append({
                    "tag_id": js.get("tag_id", p.stem),
                    "user_name": js.get("user_name", js.get("tag_id", p.stem)),
                    "protos": protos,
                })
            except Exception:
                continue
        return users

class UserMatcher:
    def __init__(self, users: List[dict]):
        self.users = users
        self.tag_ids: List[str] = []
        self.proto_mat: Optional[np.ndarray] = None
        self.proto_owner: Optional[np.ndarray] = None

        if users:
            mats = []
            owners = []
            for ui, u in enumerate(users):
                P = u["protos"]
                mats.append(P)
                owners.extend([ui] * P.shape[0])
                self.tag_ids.append(u["tag_id"])
            self.proto_mat = np.vstack(mats).astype(np.float32)
            self.proto_owner = np.array(owners, dtype=np.int32)

    def match(self, emb: np.ndarray) -> Tuple[str, float, str, float]:
        if emb is None or self.proto_mat is None or self.proto_owner is None:
            return ("unknown", -1.0, "unknown", -1.0)

        sims = self.proto_mat @ emb
        n_users = len(self.tag_ids)
        user_best = np.full((n_users,), -1.0, dtype=np.float32)

        owners = self.proto_owner
        for i in range(sims.shape[0]):
            ui = int(owners[i])
            s = float(sims[i])
            if s > user_best[ui]:
                user_best[ui] = s

        order = np.argsort(user_best)[::-1]
        best_i = int(order[0]) if order.size else -1
        second_i = int(order[1]) if order.size > 1 else -1

        best_id = self.tag_ids[best_i] if best_i >= 0 else "unknown"
        best_sim = float(user_best[best_i]) if best_i >= 0 else -1.0
        second_id = self.tag_ids[second_i] if second_i >= 0 else "unknown"
        second_sim = float(user_best[second_i]) if second_i >= 0 else -1.0

        return best_id, best_sim, second_id, second_sim


# ---------------------------
# Tracking + stability
# ---------------------------

@dataclass
class TrackState:
    track_id: int
    cx: float
    cy: float
    box: List[float]
    score: float
    last_seen: float

    best_id: str = "unknown"
    best_sim: float = -1.0
    second_id: str = "unknown"
    second_sim: float = -1.0
    margin: float = -1.0

    accepted_id: str = "unknown"
    pending_id: str = "unknown"
    pending_count: int = 0
    stable_id: str = "unknown"
    stable_committed_at: float = 0.0

class SimpleTracker:
    def __init__(self, assign_dist_px: float, ttl_sec: float):
        self.assign_dist_px = float(assign_dist_px)
        self.ttl_sec = float(ttl_sec)
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1

    def update(self, dets: List[DetOut], now: float) -> List[TrackState]:
        centers = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))

        pairs = []
        for tid, tr in self.tracks.items():
            for di, (cx, cy) in enumerate(centers):
                dist = math.hypot(cx - tr.cx, cy - tr.cy)
                pairs.append((dist, tid, di))
        pairs.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_dets = set()

        for dist, tid, di in pairs:
            if dist > self.assign_dist_px:
                continue
            if tid in assigned_tracks or di in assigned_dets:
                continue
            d = dets[di]
            cx, cy = centers[di]
            tr = self.tracks[tid]
            tr.cx, tr.cy = cx, cy
            tr.box = d.box
            tr.score = d.score
            tr.last_seen = now
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        for di, d in enumerate(dets):
            if di in assigned_dets:
                continue
            cx, cy = centers[di]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                cx=cx,
                cy=cy,
                box=d.box,
                score=d.score,
                last_seen=now,
            )

        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid].last_seen > self.ttl_sec:
                del self.tracks[tid]

        return list(self.tracks.values())

def update_stability(tr: TrackState, accepted_id: str, stable_n: int):
    tr.accepted_id = accepted_id

    if accepted_id == "unknown":
        tr.pending_id = "unknown"
        tr.pending_count = 0
        return

    if accepted_id == tr.pending_id:
        tr.pending_count += 1
    else:
        tr.pending_id = accepted_id
        tr.pending_count = 1

    if tr.pending_count >= stable_n:
        if tr.stable_id != accepted_id:
            tr.stable_id = accepted_id
            tr.stable_committed_at = time.time()


# ---------------------------
# Capture: newest-frame processing (avoid backlog)
# ---------------------------

def set_camera_low_buffer(cap: cv2.VideoCapture):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def read_latest_frame(cap: cv2.VideoCapture, max_drain: int) -> Tuple[bool, Optional[np.ndarray], int]:
    """
    Grabs and discards up to max_drain frames, then retrieves one.
    Returns: ok, frame, drained_count
    """
    drained = 0
    if max_drain > 0:
        for _ in range(max_drain):
            ok = cap.grab()
            if not ok:
                break
            drained += 1
        ok, frame = cap.retrieve()
        if ok:
            return True, frame, drained

    ok, frame = cap.read()
    if not ok:
        return False, None, drained
    return True, frame, drained


# ---------------------------
# Control state + Tk UI
# ---------------------------

class ControlState:
    def __init__(self, lights: List[str], dists: List[str]):
        self.lock = threading.Lock()

        self.lights = lights[:]
        self.dists = dists[:]

        self.light = lights[0] if lights else "Bright"
        self.dist = dists[0] if dists else "Near"

        self.roster_str = ""  # comma list tag_ids left-to-right

        self.sim_thresh = SIM_THRESHOLD_DEFAULT
        self.margin_min = MARGIN_MIN_DEFAULT
        self.stable_n = STABLE_N_DEFAULT

        self.reg_seconds = REG_SECONDS_DEFAULT
        self.trial_seconds = TRIAL_SECONDS_DEFAULT

        self.running = True

        # commands (set by UI, consumed by loop)
        self.cmd_enroll = False
        self.cmd_reload_db = False
        self.cmd_start_trial = False
        self.cmd_stop_trial = False

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "light": self.light,
                "dist": self.dist,
                "roster_str": self.roster_str,
                "sim_thresh": float(self.sim_thresh),
                "margin_min": float(self.margin_min),
                "stable_n": int(self.stable_n),
                "reg_seconds": float(self.reg_seconds),
                "trial_seconds": float(self.trial_seconds),
                "running": bool(self.running),
                "cmd_enroll": bool(self.cmd_enroll),
                "cmd_reload_db": bool(self.cmd_reload_db),
                "cmd_start_trial": bool(self.cmd_start_trial),
                "cmd_stop_trial": bool(self.cmd_stop_trial),
            }

    def consume_cmds(self) -> dict:
        with self.lock:
            cmds = {
                "enroll": self.cmd_enroll,
                "reload": self.cmd_reload_db,
                "start_trial": self.cmd_start_trial,
                "stop_trial": self.cmd_stop_trial,
            }
            self.cmd_enroll = False
            self.cmd_reload_db = False
            self.cmd_start_trial = False
            self.cmd_stop_trial = False
            return cmds


def build_ui(root: tk.Tk, state: ControlState, db_dir: Path):
    root.title("PINCH Control Panel")
    root.geometry("420x420")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="PINCH Robustness Control", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))

    # Lighting / Distance dropdowns
    ttk.Label(frm, text="Lighting").pack(anchor="w")
    light_var = tk.StringVar(value=state.light)
    light_cb = ttk.Combobox(frm, textvariable=light_var, values=state.lights, state="readonly")
    light_cb.pack(fill="x", pady=(0, 8))

    ttk.Label(frm, text="Distance").pack(anchor="w")
    dist_var = tk.StringVar(value=state.dist)
    dist_cb = ttk.Combobox(frm, textvariable=dist_var, values=state.dists, state="readonly")
    dist_cb.pack(fill="x", pady=(0, 10))

    # Roster
    ttk.Label(frm, text="Roster left-to-right (comma-separated tag_ids)").pack(anchor="w")
    roster_var = tk.StringVar(value=state.roster_str)
    roster_entry = ttk.Entry(frm, textvariable=roster_var)
    roster_entry.pack(fill="x", pady=(0, 6))

    roster_hint = ttk.Label(frm, text="Tip: click Reload DB to see registered IDs in the console.", foreground="#666")
    roster_hint.pack(anchor="w", pady=(0, 10))

    # Settings
    set_grid = ttk.Frame(frm)
    set_grid.pack(fill="x", pady=(0, 10))

    def add_labeled_spin(parent, label, var, from_, to_, step, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        sp = ttk.Spinbox(parent, textvariable=var, from_=from_, to=to_, increment=step)
        sp.grid(row=row, column=1, sticky="ew", pady=4)
        parent.grid_columnconfigure(1, weight=1)

    sim_var = tk.DoubleVar(value=state.sim_thresh)
    mar_var = tk.DoubleVar(value=state.margin_min)
    stab_var = tk.IntVar(value=state.stable_n)
    reg_var = tk.DoubleVar(value=state.reg_seconds)
    trial_var = tk.DoubleVar(value=state.trial_seconds)

    add_labeled_spin(set_grid, "sim_thresh", sim_var, 0.0, 1.0, 0.01, 0)
    add_labeled_spin(set_grid, "margin_min", mar_var, 0.0, 0.5, 0.01, 1)
    add_labeled_spin(set_grid, "stable_n", stab_var, 1, 12, 1, 2)
    add_labeled_spin(set_grid, "reg_seconds", reg_var, 3.0, 30.0, 1.0, 3)
    add_labeled_spin(set_grid, "trial_seconds", trial_var, 5.0, 120.0, 5.0, 4)

    # Buttons
    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(10, 0))

    def sync_state():
        with state.lock:
            state.light = light_var.get()
            state.dist = dist_var.get()
            state.roster_str = roster_var.get().strip()
            state.sim_thresh = float(sim_var.get())
            state.margin_min = float(mar_var.get())
            state.stable_n = int(stab_var.get())
            state.reg_seconds = float(reg_var.get())
            state.trial_seconds = float(trial_var.get())

    def cmd_enroll():
        sync_state()
        with state.lock:
            state.cmd_enroll = True

    def cmd_reload():
        sync_state()
        with state.lock:
            state.cmd_reload_db = True

    def cmd_start():
        sync_state()
        with state.lock:
            state.cmd_start_trial = True

    def cmd_stop():
        sync_state()
        with state.lock:
            state.cmd_stop_trial = True

    def cmd_quit():
        sync_state()
        with state.lock:
            state.running = False
        root.destroy()

    ttk.Button(btns, text="Enroll", command=cmd_enroll).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
    ttk.Button(btns, text="Reload DB", command=cmd_reload).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(btns, text="Start Trial", command=cmd_start).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
    ttk.Button(btns, text="Stop Trial", command=cmd_stop).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
    ttk.Button(btns, text="Quit", command=cmd_quit).grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=10)

    for c in range(2):
        btns.grid_columnconfigure(c, weight=1)

    # auto sync on dropdown changes
    def on_change(_evt=None):
        sync_state()

    light_cb.bind("<<ComboboxSelected>>", on_change)
    dist_cb.bind("<<ComboboxSelected>>", on_change)
    roster_entry.bind("<KeyRelease>", on_change)

    ttk.Separator(frm).pack(fill="x", pady=10)
    ttk.Label(frm, text=f"DB folder: {db_dir}", foreground="#666").pack(anchor="w")


# ---------------------------
# Enrollment and Trials
# ---------------------------

def enroll_user_from_ui(
    parent: tk.Tk,
    cap: cv2.VideoCapture,
    flip: bool,
    detector_backend,
    embedder: HybridPatchEmbedder,
    db: UserGestureDB,
    reg_seconds: float,
    proto_k: int,
    proto_dup_sim: float,
    max_drain: int,
):
    messagebox.showinfo(
        "Enroll",
        "Enrollment is one participant at a time.\n\n"
        "Only ONE marker should be visible.\n"
        "Hold still, then small left-right motion, then gentle tilt/roll.\n"
        "Try to move slowly to avoid blur."
    )

    user_name = simpledialog.askstring("Enroll", "Participant label (example P1):", parent=parent)
    if not user_name:
        return
    tag_id = simpledialog.askstring("Enroll", "tag_id (blank uses label):", parent=parent)
    if not tag_id:
        tag_id = user_name

    phases = [(2.0, "Hold still"), (3.0, "Small left-right motion"), (3.0, "Gentle roll/tilt"), (2.0, "Hold still")]
    start = time.time()
    embeds: List[np.ndarray] = []

    win = "PINCH Enroll (close window to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame, drained = read_latest_frame(cap, max_drain=max_drain)
        if not ok or frame is None:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        elapsed = time.time() - start
        if elapsed >= reg_seconds:
            break

        # phase label
        ph_label = "Move"
        t = elapsed
        for dur, label in phases:
            if t <= dur:
                ph_label = label
                break
            t -= dur

        td0 = time.time()
        dets = detector_backend.detect(frame)
        td1 = time.time()
        det_ms = (td1 - td0) * 1000.0

        if dets:
            best = dets[0]
            x1, y1, x2, y2 = map(int, best.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            te0 = time.time()
            emb = embedder.embed(frame, best.box)
            te1 = time.time()
            emb_ms = (te1 - te0) * 1000.0

            if emb is not None:
                embeds.append(emb)

            cv2.putText(
                frame,
                f"samples={len(embeds)} det_ms={det_ms:.1f} emb_ms={emb_ms:.1f} drained={drained}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(frame, "No marker detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"{user_name}  tag_id={tag_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"phase: {ph_label}   t={elapsed:.1f}/{reg_seconds:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        # if user closed the window
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win)

    if len(embeds) < 10:
        messagebox.showwarning("Enroll", "Too few samples. Move closer, slow down, and try again.")
        return

    db.save_user(tag_id=tag_id, user_name=user_name, embeds=embeds, k=proto_k, dup_sim=proto_dup_sim)
    messagebox.showinfo("Enroll", f"Saved {tag_id} with {len(embeds)} samples and multi-prototypes.")


@dataclass
class TrialEventState:
    # slot-level state for event logging
    last_stable_id: str = "unknown"
    miss_active: bool = False
    miss_start_t: float = 0.0
    last_commit_logged_t: float = 0.0

def run_trial(
    cap: cv2.VideoCapture,
    flip: bool,
    detector_backend,
    use_backend_track_ids: bool,
    embedder: HybridPatchEmbedder,
    matcher: UserMatcher,
    db_users_count: int,
    roster: List[str],
    lighting: str,
    distance: str,
    sim_thresh: float,
    margin_min: float,
    stable_n: int,
    trial_seconds: float,
    session_dir: Path,
    proto_info: str,
    max_drain: int,
    stop_flag_fn,
):
    """
    Writes:
    - *_per_frame.csv
    - *_summary.json
    - *_events.jsonl (time-to-stable and re-id events)
    """

    if not roster:
        return

    trial_id = f"{lighting}_{distance}_{now_stamp()}"
    per_frame_path = session_dir / f"{trial_id}_per_frame.csv"
    summary_path = session_dir / f"{trial_id}_summary.json"
    events_path = session_dir / f"{trial_id}_events.jsonl"

    N = len(roster)

    # set up tracker if backend doesn't provide ids
    tracker = None if use_backend_track_ids else SimpleTracker(TRACK_ASSIGN_DIST_PX, TRACK_TTL_SEC)

    # per-slot counters
    slot_total = [0] * N
    slot_miss = [0] * N
    slot_wrong_acc = [0] * N
    slot_correct_acc = [0] * N
    slot_correct_stable = [0] * N

    # latency arrays
    lat_det = []
    lat_total = []
    fps_proc = []
    drained_frames = []

    # slot event state
    slot_evt: List[TrialEventState] = [TrialEventState() for _ in range(N)]
    reid_gap_s = 0.25

    cols = [
        "trial_id","t_wall","frame_idx","lighting","distance",
        "n_users_roster","n_users_db",
        "cam_w","cam_h","proc_fps","drained",
        "slot_idx","true_id",
        "track_id","detected","det_score",
        "bbox_x1","bbox_y1","bbox_x2","bbox_y2","bbox_area",
        "best_id","best_sim","second_id","second_sim","margin",
        "accepted_id","stable_id","pending_id","pending_count",
        "is_correct_accepted","is_correct_stable","is_wrong_accepted","is_miss",
        "lat_det_ms","lat_emb_ms","lat_match_ms","lat_total_ms",
    ]

    win = "PINCH Trial (press ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_start = time.time()
    prev_loop_t = t_start
    frame_idx = 0

    with open(per_frame_path, "w", encoding="utf-8") as fp, open(events_path, "w", encoding="utf-8") as fev:
        write_csv_header(fp, cols)

        while True:
            if stop_flag_fn():
                break

            ok, frame, drained = read_latest_frame(cap, max_drain=max_drain)
            if not ok or frame is None:
                break

            if flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            elapsed = now - t_start
            if elapsed >= trial_seconds:
                break

            frame_idx += 1
            H, W = frame.shape[:2]
            slot_w = W / float(N)

            # processed fps estimate
            dt_loop = max(1e-6, now - prev_loop_t)
            prev_loop_t = now
            fps_now = 1.0 / dt_loop

            # detection timing
            td0 = time.time()
            dets = detector_backend.detect(frame)
            td1 = time.time()
            det_ms = (td1 - td0) * 1000.0

            # turn detections into TrackState list
            tracks: List[TrackState] = []
            if use_backend_track_ids:
                # use det_id as track_id (ByteTrack)
                for d in dets:
                    if d.det_id is None:
                        continue
                    x1, y1, x2, y2 = d.box
                    tracks.append(TrackState(
                        track_id=int(d.det_id),
                        cx=(x1 + x2) * 0.5,
                        cy=(y1 + y2) * 0.5,
                        box=d.box,
                        score=d.score,
                        last_seen=now,
                    ))
            else:
                # simple tracker
                assert tracker is not None
                tracks = tracker.update(dets, now=now)

            # embed + match + stability
            emb_ms_sum = 0.0
            match_ms_sum = 0.0
            for tr in tracks:
                te0 = time.time()
                emb = embedder.embed(frame, tr.box)
                te1 = time.time()
                emb_ms_sum += (te1 - te0) * 1000.0

                tm0 = time.time()
                best_id, best_sim, second_id, second_sim = matcher.match(emb)
                tm1 = time.time()
                match_ms_sum += (tm1 - tm0) * 1000.0

                tr.best_id = best_id
                tr.best_sim = float(best_sim)
                tr.second_id = second_id
                tr.second_sim = float(second_sim)
                tr.margin = float(best_sim - second_sim) if (best_sim >= 0 and second_sim >= 0) else -1.0

                if (tr.best_sim >= sim_thresh) and (tr.margin >= margin_min):
                    accepted = tr.best_id
                else:
                    accepted = "unknown"

                update_stability(tr, accepted_id=accepted, stable_n=stable_n)

            total_ms = det_ms + emb_ms_sum + match_ms_sum

            lat_det.append(det_ms)
            lat_total.append(total_ms)
            fps_proc.append(fps_now)
            drained_frames.append(float(drained))

            # assign best track per slot
            best_track_per_slot: List[Optional[TrackState]] = [None] * N
            for tr in tracks:
                si = int(tr.cx / slot_w)
                si = max(0, min(N - 1, si))
                prev = best_track_per_slot[si]
                if prev is None or tr.score > prev.score:
                    best_track_per_slot[si] = tr

            # log per slot rows
            for si in range(N):
                true_id = roster[si]
                slot_total[si] += 1
                tr = best_track_per_slot[si]

                if tr is None:
                    slot_miss[si] += 1

                    # event: begin miss period if previously stable
                    evt = slot_evt[si]
                    if (evt.last_stable_id != "unknown") and (not evt.miss_active):
                        evt.miss_active = True
                        evt.miss_start_t = now

                    write_csv_row(fp, cols, {
                        "trial_id": trial_id, "t_wall": now, "frame_idx": frame_idx,
                        "lighting": lighting, "distance": distance,
                        "n_users_roster": N, "n_users_db": db_users_count,
                        "cam_w": W, "cam_h": H, "proc_fps": fps_now, "drained": drained,
                        "slot_idx": si, "true_id": true_id,
                        "track_id": -1, "detected": 0, "det_score": -1.0,
                        "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1, "bbox_area": 0.0,
                        "best_id": "unknown", "best_sim": -1.0, "second_id": "unknown", "second_sim": -1.0, "margin": -1.0,
                        "accepted_id": "unknown", "stable_id": "unknown", "pending_id": "unknown", "pending_count": 0,
                        "is_correct_accepted": 0, "is_correct_stable": 0, "is_wrong_accepted": 0, "is_miss": 1,
                        "lat_det_ms": det_ms, "lat_emb_ms": 0.0, "lat_match_ms": 0.0, "lat_total_ms": det_ms,
                    })
                    continue

                x1, y1, x2, y2 = tr.box
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

                accepted_id = tr.accepted_id
                stable_id = tr.stable_id

                is_correct_accepted = 1 if (accepted_id == true_id) else 0
                is_correct_stable = 1 if (stable_id == true_id) else 0
                is_wrong_accepted = 1 if (accepted_id != "unknown" and accepted_id != true_id) else 0

                if is_correct_accepted:
                    slot_correct_acc[si] += 1
                elif is_wrong_accepted:
                    slot_wrong_acc[si] += 1

                if is_correct_stable:
                    slot_correct_stable[si] += 1

                # events: stable commit and re-id
                evt = slot_evt[si]
                # stable commit event, log only when stable changes
                if stable_id != evt.last_stable_id:
                    # time-to-stable: we can approximate as "time since trial start" when stable first appears
                    # (you can compute tighter offline if you want)
                    if stable_id != "unknown":
                        ev = {
                            "event": "stable_commit",
                            "trial_id": trial_id,
                            "t_wall": now,
                            "slot_idx": si,
                            "true_id": true_id,
                            "stable_id": stable_id,
                            "t_since_trial_s": float(elapsed),
                        }
                        fev.write(json.dumps(ev) + "\n")

                        # re-id event if we were in a miss gap
                        if evt.miss_active and (now - evt.miss_start_t) >= reid_gap_s and (stable_id == evt.last_stable_id or evt.last_stable_id == "unknown"):
                            ev2 = {
                                "event": "reid",
                                "trial_id": trial_id,
                                "t_wall": now,
                                "slot_idx": si,
                                "true_id": true_id,
                                "stable_id": stable_id,
                                "reid_time_s": float(now - evt.miss_start_t),
                            }
                            fev.write(json.dumps(ev2) + "\n")
                        evt.miss_active = False

                    evt.last_stable_id = stable_id

                # if we see a detection, miss period ends only after a stable appears again
                # but we keep miss_active until stable commit, so reid_time is "miss_start -> stable_commit"

                write_csv_row(fp, cols, {
                    "trial_id": trial_id, "t_wall": now, "frame_idx": frame_idx,
                    "lighting": lighting, "distance": distance,
                    "n_users_roster": N, "n_users_db": db_users_count,
                    "cam_w": W, "cam_h": H, "proc_fps": fps_now, "drained": drained,
                    "slot_idx": si, "true_id": true_id,
                    "track_id": tr.track_id, "detected": 1, "det_score": tr.score,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2, "bbox_area": area,
                    "best_id": tr.best_id, "best_sim": tr.best_sim, "second_id": tr.second_id, "second_sim": tr.second_sim, "margin": tr.margin,
                    "accepted_id": accepted_id, "stable_id": stable_id, "pending_id": tr.pending_id, "pending_count": tr.pending_count,
                    "is_correct_accepted": is_correct_accepted, "is_correct_stable": is_correct_stable,
                    "is_wrong_accepted": is_wrong_accepted, "is_miss": 0,
                    "lat_det_ms": det_ms, "lat_emb_ms": emb_ms_sum, "lat_match_ms": match_ms_sum, "lat_total_ms": total_ms,
                })

            # draw UI
            for si in range(1, N):
                x = int(si * slot_w)
                cv2.line(frame, (x, 0), (x, H), (80, 80, 80), 1)

            draw_tracks = sorted(tracks, key=lambda t: t.score, reverse=True)[:MAX_DRAW_TRACKS]
            for tr in draw_tracks:
                x1i, y1i, x2i, y2i = map(int, tr.box)
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (120, 255, 120), 2)
                cv2.putText(
                    frame,
                    f"T{tr.track_id} st={tr.stable_id}",
                    (x1i, max(18, y1i - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 120),
                    2,
                )

            cv2.putText(frame, f"{trial_id}  t={elapsed:.1f}/{trial_seconds:.1f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Roster: {', '.join(roster)}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"proc_fps={fps_now:.1f} drained={drained} total_ms~{total_ms:.1f}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, proto_info, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyWindow(win)

    # summary
    lat_det_arr = np.array(lat_det, dtype=np.float32) if lat_det else np.array([0.0], dtype=np.float32)
    lat_total_arr = np.array(lat_total, dtype=np.float32) if lat_total else np.array([0.0], dtype=np.float32)
    fps_arr = np.array(fps_proc, dtype=np.float32) if fps_proc else np.array([0.0], dtype=np.float32)
    drained_arr = np.array(drained_frames, dtype=np.float32) if drained_frames else np.array([0.0], dtype=np.float32)

    per_slot = []
    for si in range(N):
        total = slot_total[si]
        per_slot.append({
            "slot_idx": si,
            "true_id": roster[si],
            "frames_total": int(total),
            "miss_rate": safe_div(slot_miss[si], total),
            "correct_rate_accepted": safe_div(slot_correct_acc[si], total),
            "wrong_rate_accepted": safe_div(slot_wrong_acc[si], total),
            "correct_rate_stable": safe_div(slot_correct_stable[si], total),
        })

    summary = {
        "trial_id": trial_id,
        "lighting": lighting,
        "distance": distance,
        "trial_seconds": float(trial_seconds),
        "roster": roster,
        "sim_thresh": float(sim_thresh),
        "margin_min": float(margin_min),
        "stable_n": int(stable_n),
        "per_frame_csv": str(per_frame_path),
        "events_jsonl": str(events_path),
        "lat_det_ms_median": float(np.median(lat_det_arr)),
        "lat_total_ms_median": float(np.median(lat_total_arr)),
        "lat_total_ms_p95": float(np.percentile(lat_total_arr, 95)),
        "proc_fps_median": float(np.median(fps_arr)),
        "drained_frames_median": float(np.median(drained_arr)),
        "per_slot": per_slot,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return trial_id, str(per_frame_path), str(summary_path), str(events_path)


# ---------------------------
# Main loop (OpenCV + Tk)
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--db-dir", type=str, default="user_gestures")
    ap.add_argument("--out-dir", type=str, default="logs")
    ap.add_argument("--max-drain", type=int, default=2, help="How many frames to grab+discard each loop (reduce webcam backlog). 0 disables.")
    ap.add_argument("--tflite", type=str, default="", help="Path to .tflite (optional)")
    ap.add_argument("--pt", type=str, default="", help="Path to .pt (optional, uses ByteTrack if ultralytics installed)")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF_DEFAULT)
    ap.add_argument("--proto-k", type=int, default=PROTO_K_DEFAULT)
    ap.add_argument("--proto-dup-sim", type=float, default=PROTO_DUP_SIM_DEFAULT)
    ap.add_argument("--lights", type=str, default=",".join(DEFAULT_LIGHTS))
    ap.add_argument("--distances", type=str, default=",".join(DEFAULT_DISTANCES))
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parent
    db_dir = root_dir / args.db_dir
    out_root = root_dir / args.out_dir
    ensure_dir(db_dir)
    ensure_dir(out_root)

    lights = [x.strip() for x in args.lights.split(",") if x.strip()] or DEFAULT_LIGHTS[:]
    dists = [x.strip() for x in args.distances.split(",") if x.strip()] or DEFAULT_DISTANCES[:]

    flip = not args.no_flip
    max_drain = max(0, int(args.max_drain))

    # pick backend
    detector_backend = None
    use_backend_track_ids = False
    model_desc = ""

    if args.pt.strip():
        pt_path = args.pt.strip()
        if not Path(pt_path).exists():
            raise FileNotFoundError(f".pt not found: {pt_path}")
        try:
            detector_backend = UltralyticsYoloTracker(pt_path, tracker_yaml="bytetrack.yaml", conf=args.min_conf, iou=0.5)
            use_backend_track_ids = True
            model_desc = f"PT+ByteTrack: {Path(pt_path).name}"
        except Exception as e:
            raise RuntimeError(f"Failed to init ultralytics backend: {e}")

    else:
        tflite_path = args.tflite.strip() or auto_find_model(root_dir, (".tflite",))
        if not tflite_path or not Path(tflite_path).exists():
            raise FileNotFoundError("No .tflite found. Put one beside this script or in ./assets, or pass --tflite.")
        detector_backend = TFLiteYoloDetector(tflite_path, num_threads=2, min_conf=args.min_conf)
        use_backend_track_ids = False
        model_desc = f"TFLite: {Path(tflite_path).name}"

    embedder = HybridPatchEmbedder()
    db = UserGestureDB(db_dir)

    # session dir
    session_id = f"session_{now_stamp()}"
    session_dir = out_root / session_id
    ensure_dir(session_dir)

    # save session meta
    session_meta = {
        "session_id": session_id,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_desc": model_desc,
        "min_conf": float(args.min_conf),
        "proto_k": int(args.proto_k),
        "proto_dup_sim": float(args.proto_dup_sim),
        "flip": bool(flip),
        "max_drain": int(max_drain),
        "lights": lights,
        "distances": dists,
    }
    with open(session_dir / "session_meta.json", "w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=2)

    # camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")
    set_camera_low_buffer(cap)

    # state + UI
    state = ControlState(lights=lights, dists=dists)

    ui_root = tk.Tk()
    build_ui(ui_root, state, db_dir)

    # show console helpful info
    print("\nPINCH UI started.")
    print(f"Session logs: {session_dir}")
    print("Enrolled IDs live in:", db_dir)
    print("If you click Reload DB, I will print the known tag_ids here.\n")

    # runtime matcher
    users = db.load_all()
    matcher = UserMatcher(users)

    # stop flag for trial
    stop_trial_flag = {"stop": False}

    def stop_flag_fn():
        return stop_trial_flag["stop"]

    # main processing loop driven by Tk timer
    win_preview = "PINCH Preview"
    cv2.namedWindow(win_preview, cv2.WINDOW_NORMAL)

    def loop_tick():
        snap = state.snapshot()
        if not snap["running"]:
            cap.release()
            cv2.destroyAllWindows()
            return

        cmds = state.consume_cmds()

        # reload DB
        nonlocal_users = False
        if cmds["reload"]:
            users_local = db.load_all()
            matcher_local = UserMatcher(users_local)
            # update outer variables (python closure workaround)
            nonlocal users, matcher
            users = users_local
            matcher = matcher_local
            print("[DB] Registered tag_ids:", [u["tag_id"] for u in users])

        # enroll
        if cmds["enroll"]:
            enroll_user_from_ui(
                parent=ui_root,
                cap=cap,
                flip=flip,
                detector_backend=detector_backend,
                embedder=embedder,
                db=db,
                reg_seconds=float(snap["reg_seconds"]),
                proto_k=int(args.proto_k),
                proto_dup_sim=float(args.proto_dup_sim),
                max_drain=max_drain,
            )
            users_local = db.load_all()
            matcher_local = UserMatcher(users_local)
            users = users_local
            matcher = matcher_local
            print("[DB] Registered tag_ids:", [u["tag_id"] for u in users])

        # start trial
        if cmds["start_trial"]:
            roster_str = snap["roster_str"]
            roster = [x.strip() for x in roster_str.split(",") if x.strip()]
            if not roster:
                messagebox.showwarning("Start Trial", "Roster is empty. Enter comma-separated tag_ids left-to-right.")
            elif len(matcher.tag_ids) == 0:
                messagebox.showwarning("Start Trial", "No enrolled users. Enroll first.")
            else:
                stop_trial_flag["stop"] = False
                proto_info = f"{model_desc} | K={args.proto_k} dup<{args.proto_dup_sim:.2f} | stable_n={int(snap['stable_n'])}"
                out = run_trial(
                    cap=cap,
                    flip=flip,
                    detector_backend=detector_backend,
                    use_backend_track_ids=use_backend_track_ids,
                    embedder=embedder,
                    matcher=matcher,
                    db_users_count=len(matcher.tag_ids),
                    roster=roster,
                    lighting=snap["light"],
                    distance=snap["dist"],
                    sim_thresh=float(snap["sim_thresh"]),
                    margin_min=float(snap["margin_min"]),
                    stable_n=int(snap["stable_n"]),
                    trial_seconds=float(snap["trial_seconds"]),
                    session_dir=session_dir,
                    proto_info=proto_info,
                    max_drain=max_drain,
                    stop_flag_fn=stop_flag_fn,
                )
                if out:
                    trial_id, csvp, sump, evp = out
                    messagebox.showinfo("Trial Saved", f"{trial_id}\n\nCSV:\n{csvp}\n\nSummary:\n{sump}\n\nEvents:\n{evp}")

        # stop trial
        if cmds["stop_trial"]:
            stop_trial_flag["stop"] = True

        # preview frame
        ok, frame, drained = read_latest_frame(cap, max_drain=max_drain)
        if ok and frame is not None:
            if flip:
                frame = cv2.flip(frame, 1)

            # light preview: detect + draw boxes, show stable id if possible
            td0 = time.time()
            dets = detector_backend.detect(frame)
            td1 = time.time()
            det_ms = (td1 - td0) * 1000.0

            # build track list for preview
            tracks_preview: List[TrackState] = []
            if use_backend_track_ids:
                for d in dets[:MAX_DRAW_TRACKS]:
                    if d.det_id is None:
                        continue
                    x1, y1, x2, y2 = d.box
                    tracks_preview.append(TrackState(
                        track_id=int(d.det_id),
                        cx=(x1 + x2) * 0.5,
                        cy=(y1 + y2) * 0.5,
                        box=d.box,
                        score=d.score,
                        last_seen=time.time(),
                    ))
            else:
                # preview without a persistent tracker to keep it simple
                # (trial uses a tracker when needed)
                for i, d in enumerate(dets[:MAX_DRAW_TRACKS]):
                    x1, y1, x2, y2 = d.box
                    tracks_preview.append(TrackState(
                        track_id=i + 1,
                        cx=(x1 + x2) * 0.5,
                        cy=(y1 + y2) * 0.5,
                        box=d.box,
                        score=d.score,
                        last_seen=time.time(),
                    ))

            # match preview (optional)
            if len(matcher.tag_ids) > 0:
                for tr in tracks_preview:
                    emb = embedder.embed(frame, tr.box)
                    best_id, best_sim, second_id, second_sim = matcher.match(emb)
                    margin = best_sim - second_sim if (best_sim >= 0 and second_sim >= 0) else -1.0
                    if (best_sim >= float(snap["sim_thresh"])) and (margin >= float(snap["margin_min"])):
                        accepted = best_id
                    else:
                        accepted = "unknown"
                    update_stability(tr, accepted_id=accepted, stable_n=int(snap["stable_n"]))

            for tr in tracks_preview:
                x1, y1, x2, y2 = map(int, tr.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    frame,
                    f"{tr.stable_id}  det={tr.score:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    2,
                )

            cv2.putText(frame, f"{model_desc}  det_ms={det_ms:.1f} drained={drained}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Lighting={snap['light']}  Distance={snap['dist']}", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"DB users={len(matcher.tag_ids)}  Roster={snap['roster_str'] or '[empty]'}", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(win_preview, frame)
            cv2.waitKey(1)

        # schedule next tick
        ui_root.after(1, loop_tick)

    ui_root.after(1, loop_tick)
    ui_root.mainloop()

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
