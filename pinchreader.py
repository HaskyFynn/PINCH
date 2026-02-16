"""
PINCH Live System (single script) - Click UI + Enrollment + Trials + Logging

- Pure OpenCV UI (mouse clicks for menus, toggles, GT assignment)
- Enrollment: guided steps, builds prototypes from embeddings
- Trials: YOLO + ByteTrack + embedder + prototype match per track
- Logging: minimal paper metrics (latency, track loss, switches, accuracy if GT)
- No tkinter

Dependencies:
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow
"""

import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# USER SETTINGS (EDIT THESE)
# ============================================================
YOLO_WEIGHTS = r"C:\path\to\yolo_best.pt"
EMBEDDER_WEIGHTS = r"C:\path\to\embedder_resnet18_triplet.pt"
RUN_DIR = r"C:\path\to\runs\pinch_live"

USE_WEBCAM = True
WEBCAM_INDEX = 0
VIDEO_PATH = r""  # used if USE_WEBCAM=False

MAX_MARKERS = 4

# UI canvas size (fixed to keep mouse mapping correct)
CANVAS_W = 1280
CANVAS_H = 720

# Detection + tracking
DET_CONF = 0.50
DET_IOU = 0.70
TRACKER_YAML = "bytetrack.yaml"

# Crop gating
MIN_BOX_AREA = 40 * 40
BLUR_THRES = 25.0
BOX_PAD_FRAC = 0.06

# Embedder
EMBED_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# Prototypes
PROTOS_PER_MARKER = 5
PROTOS_KMEANS_ITERS = 18
THRESH_PERCENTILE = 5  # marker threshold = this percentile of enrollment sims

# Track identity smoothing
EMA_ALPHA = 0.70
DECISION_HYST = 0.05

# Enrollment guidance (seconds)
ENROLL_STEPS = [
    ("Hold front-facing", 8),
    ("Swipe left-right", 10),
    ("Swipe up-down", 10),
    ("Hold left side", 6),
    ("Hold right side", 6),
    ("Tilt + motion", 8),
]

# Trial durations (seconds)
TRIAL_SWIPE_SEC = 12
TRIAL_INTERFERE_SEC = 18
TRIAL_REENTRY_SEC = 18

# ============================================================
# UI colors (modern dark)
# ============================================================
BG = (22, 22, 26)
PANEL = (28, 28, 34)
ACCENT = (0, 180, 255)
ACCENT2 = (0, 220, 255)
TEXT = (230, 230, 235)
MUTED = (160, 160, 170)
OK = (0, 255, 100)
WARN = (255, 80, 80)
UNKNOWN = (0, 150, 255)


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.60):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text(img, text, org, scale=0.62, thick=2, color=TEXT):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_rounded_rect(img, x1, y1, x2, y2, r=12, color=PANEL, thickness=-1):
    # Simple rounded rect using circles + rects
    if thickness != -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
        cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
        cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
        cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)
        return

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1)

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def lap_var(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def open_source():
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(WEBCAM_INDEX)
        return cap
    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_PATH)
    return cap


# ============================================================
# Embedder
# ============================================================
class ResnetEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = m.fc.in_features  # must read before replacing
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, p=2, dim=1)

def load_embedder(weights_path: str, device: str) -> nn.Module:
    model = ResnetEmbedder(EMBED_DIM).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

EMBED_TFM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def embed_crops(model: nn.Module, crops_rgb: List[np.ndarray], device: str) -> np.ndarray:
    if not crops_rgb:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    xs = [EMBED_TFM(Image.fromarray(c)) for c in crops_rgb]
    x = torch.stack(xs, dim=0).to(device)
    if USE_AMP and device == "cuda":
        with torch.cuda.amp.autocast():
            z = model(x)
    else:
        z = model(x)
    return z.detach().cpu().numpy().astype(np.float32)


# ============================================================
# Registry + prototypes
# ============================================================
@dataclass
class MarkerProfile:
    marker_id: str
    proto: List[List[float]]
    thr: float
    enroll_frames: int
    enroll_used: int

class Registry:
    def __init__(self):
        self.markers: List[MarkerProfile] = []

    def add_marker(self, profile: MarkerProfile):
        self.markers.append(profile)

    def names(self) -> List[str]:
        return [m.marker_id for m in self.markers]

    def to_json(self) -> Dict:
        return {"markers": [asdict(m) for m in self.markers]}

    @staticmethod
    def from_json(d: Dict) -> "Registry":
        r = Registry()
        for m in d.get("markers", []):
            r.add_marker(MarkerProfile(**m))
        return r

def kmeans_prototypes(X: np.ndarray, k: int, iters: int, seed: int = 0) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    set_seed(seed)
    Xn = np.stack([safe_norm(x) for x in X], axis=0)
    k = min(k, Xn.shape[0])
    idx = np.random.choice(Xn.shape[0], size=k, replace=False)
    C = Xn[idx].copy()
    for _ in range(iters):
        sims = Xn @ C.T
        a = np.argmax(sims, axis=1)
        for j in range(k):
            pts = Xn[a == j]
            if len(pts) > 0:
                C[j] = safe_norm(pts.mean(axis=0))
            else:
                C[j] = Xn[np.random.randint(0, Xn.shape[0])]
    return C.astype(np.float32)

def build_profile_from_enrollment(marker_id: str, embs: np.ndarray) -> MarkerProfile:
    if embs.shape[0] < 10:
        protos = kmeans_prototypes(embs, k=min(2, embs.shape[0]), iters=8, seed=0)
    else:
        protos = kmeans_prototypes(embs, k=PROTOS_PER_MARKER, iters=PROTOS_KMEANS_ITERS, seed=0)

    sims = embs @ protos.T if protos.shape[0] > 0 else np.zeros((embs.shape[0], 1), dtype=np.float32)
    best = sims.max(axis=1) if sims.shape[1] > 0 else np.zeros((embs.shape[0],), dtype=np.float32)
    thr = float(np.percentile(best, THRESH_PERCENTILE)) if best.shape[0] > 0 else 0.0

    return MarkerProfile(
        marker_id=marker_id,
        proto=protos.tolist(),
        thr=thr,
        enroll_frames=0,
        enroll_used=int(embs.shape[0]),
    )

def match_marker(emb: np.ndarray, registry: Registry) -> Tuple[str, float]:
    best_name = "unknown"
    best_sim = -1.0
    for m in registry.markers:
        P = np.array(m.proto, dtype=np.float32)
        if P.size == 0:
            continue
        sim = float(np.max(P @ emb))
        if sim > best_sim:
            best_sim = sim
            best_name = m.marker_id

    if best_name != "unknown":
        thr = None
        for m in registry.markers:
            if m.marker_id == best_name:
                thr = m.thr
                break
        if thr is not None and best_sim < thr:
            return "unknown", best_sim

    return best_name, best_sim


# ============================================================
# Track state
# ============================================================
@dataclass
class TrackState:
    track_id: int
    z_ema: Optional[np.ndarray] = None
    last_name: str = "unknown"
    last_sim: float = -1.0
    switches: int = 0
    seen_frames: int = 0
    last_seen_frame: int = -1

def update_identity(ts: TrackState, z: np.ndarray, registry: Registry) -> Tuple[str, float]:
    # update EMA embedding
    if ts.z_ema is None:
        ts.z_ema = z.copy()
    else:
        ts.z_ema = safe_norm(EMA_ALPHA * ts.z_ema + (1.0 - EMA_ALPHA) * z)

    # classify using EMA
    pred, sim = match_marker(ts.z_ema, registry)

    # hysteresis on switching
    if pred != ts.last_name:
        if sim >= ts.last_sim + DECISION_HYST:
            ts.switches += 1
            ts.last_name = pred
            ts.last_sim = sim
    else:
        ts.last_sim = max(ts.last_sim, sim)

    ts.seen_frames += 1
    return ts.last_name, ts.last_sim


# ============================================================
# Logging
# ============================================================
def new_frame_logger():
    cols = [
        "trial_id", "trial_type", "condition",
        "frame_idx", "t_ms",
        "n_dets", "n_tracks",
        "det_track_ms", "embed_ms", "match_ms", "total_ms",
        "new_tracks", "lost_tracks",
        "unknown_tracks",
    ]
    return cols, []

def new_event_logger():
    cols = [
        "trial_id", "trial_type", "condition",
        "t_ms", "event",
        "track_id", "pred", "gt", "sim",
    ]
    return cols, []

def save_confusion_matrix(classes: List[str], cm: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Marker ID Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# UI primitives
# ============================================================
@dataclass
class Button:
    text: str
    rect: Tuple[int, int, int, int]  # x,y,w,h
    enabled: bool = True
    hover: bool = False
    toggled: bool = False
    tag: str = ""

def point_in_rect(x, y, r):
    rx, ry, rw, rh = r
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def draw_button(canvas, b: Button):
    x, y, w, h = b.rect
    if not b.enabled:
        col = (70, 70, 78)
    else:
        if b.hover:
            col = ACCENT2
        else:
            col = ACCENT
    if b.toggled:
        col = (0, 210, 170)

    draw_rounded_rect(canvas, x, y, x + w, y + h, r=14, color=col, thickness=-1)
    draw_rounded_rect(canvas, x, y, x + w, y + h, r=14, color=(0, 0, 0), thickness=2)

    ts = cv2.getTextSize(b.text, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 2)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2
    put_text(canvas, b.text, (tx, ty), scale=0.66, thick=2, color=TEXT)


# ============================================================
# Canvas composition: place camera frame into fixed canvas
# ============================================================
def compose_canvas(frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """
    Returns:
    - canvas (CANVAS_H, CANVAS_W)
    - scale from frame to canvas
    - offx, offy where the resized frame is placed
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = BG

    fh, fw = frame_bgr.shape[:2]
    s = min((CANVAS_W - 40) / max(fw, 1), (CANVAS_H - 40) / max(fh, 1))
    s = max(min(s, 1.0), 1e-6)
    rw, rh = int(fw * s), int(fh * s)
    resized = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)

    offx = (CANVAS_W - rw) // 2
    offy = (CANVAS_H - rh) // 2
    canvas[offy:offy + rh, offx:offx + rw] = resized

    return canvas, s, offx, offy


# ============================================================
# Main App
# ============================================================
class PINCHApp:
    def __init__(self, yolo: YOLO, embedder: nn.Module):
        self.yolo = yolo
        self.embedder = embedder

        self.registry: Optional[Registry] = None
        self.reg_path = os.path.join(RUN_DIR, "registry.json")

        self.running = True
        self.screen = "main"  # main, enroll_setup, enroll_capture, trial_setup, trial_run, summary

        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

        # Enrollment state
        self.enroll_count = 1
        self.enroll_names: List[str] = []
        self.active_name_idx = 0
        self.name_edit = ""   # keyboard typed
        self.enroll_marker_idx = 0
        self.enroll_step_idx = 0
        self.enroll_step_t0 = None
        self.enroll_t0 = None
        self.enroll_embs: List[np.ndarray] = []
        self.enroll_frames = 0
        self.enroll_used = 0

        # Trial state
        self.trial_type = "swipe"
        self.condition_near = True
        self.condition_bright = True
        self.trial_duration = TRIAL_SWIPE_SEC
        self.trial_id = ""
        self.trial_t0 = None
        self.frame_idx = 0
        self.states: Dict[int, TrackState] = {}
        self.active_prev = set()

        # GT assignment
        self.gt_enabled = True
        self.selected_track: Optional[int] = None
        self.gt_map: Dict[int, str] = {}

        # logs
        self.frame_cols, self.frame_rows = new_frame_logger()
        self.event_cols, self.event_rows = new_event_logger()
        self.lat_ms: List[float] = []

        self.summary_data = None

        cv2.namedWindow("PINCH Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PINCH Live", CANVAS_W, CANVAS_H)
        cv2.setMouseCallback("PINCH Live", self.on_mouse)

        self.load_registry_if_any()

    def load_registry_if_any(self):
        if os.path.isfile(self.reg_path):
            try:
                with open(self.reg_path, "r", encoding="utf-8") as f:
                    self.registry = Registry.from_json(json.load(f))
                if len(self.registry.markers) == 0:
                    self.registry = None
            except:
                self.registry = None

    def save_registry(self):
        ensure_dir(RUN_DIR)
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(self.registry.to_json(), f, indent=2)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True

    def consume_click(self) -> bool:
        if self.mouse_clicked:
            self.mouse_clicked = False
            return True
        return False

    def condition_str(self) -> str:
        d = "near" if self.condition_near else "far"
        l = "bright" if self.condition_bright else "dim"
        return f"{d}/{l}"

    def handle_keyboard_for_text(self, key):
        # simple name typing (letters, digits, space, underscore, dash)
        if key in (8, 127):  # backspace
            self.name_edit = self.name_edit[:-1]
            return
        if key in (13, 10):  # enter
            return
        if 32 <= key <= 126 and len(self.name_edit) < 18:
            self.name_edit += chr(key)

    # ---------------- UI Screens ----------------

    def draw_header(self, canvas, title, subtitle=""):
        alpha_rect(canvas, 0, 0, CANVAS_W, 90, color=(0, 0, 0), alpha=0.55)
        put_text(canvas, title, (28, 40), scale=1.00, thick=2, color=ACCENT)
        if subtitle:
            put_text(canvas, subtitle, (28, 74), scale=0.55, thick=2, color=MUTED)

    def main_screen(self, canvas):
        self.draw_header(canvas, "PINCH Live", "Click to navigate. Enrollment builds prototypes. Trials log metrics for paper.")

        # registry status
        if self.registry is None:
            msg = "Registry: not found. Please enroll markers."
            col = WARN
        else:
            msg = "Registry: " + ", ".join(self.registry.names())
            col = OK
        put_text(canvas, msg, (28, 120), scale=0.62, thick=2, color=col)

        # buttons
        btns = []
        x = 440
        y0 = 180
        w = 400
        h = 62
        gap = 18

        def add(text, tag, enabled=True):
            b = Button(text=text, rect=(x, y0 + len(btns) * (h + gap), w, h), enabled=enabled, tag=tag)
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            btns.append(b)

        add("Enroll Markers", "enroll", enabled=True)
        add("Swipe Trial", "trial_swipe", enabled=(self.registry is not None))
        add("Interference Trial", "trial_interfere", enabled=(self.registry is not None))
        add("Re-entry Trial", "trial_reentry", enabled=(self.registry is not None))
        add("Exit", "exit", enabled=True)

        for b in btns:
            draw_button(canvas, b)

        if self.consume_click():
            for b in btns:
                if b.enabled and b.hover:
                    if b.tag == "enroll":
                        self.start_enroll_setup()
                    elif b.tag == "trial_swipe":
                        self.start_trial_setup("swipe")
                    elif b.tag == "trial_interfere":
                        self.start_trial_setup("interfere")
                    elif b.tag == "trial_reentry":
                        self.start_trial_setup("reentry")
                    elif b.tag == "exit":
                        self.running = False
                    break

    def start_enroll_setup(self):
        self.screen = "enroll_setup"
        self.enroll_count = 1
        self.enroll_names = []
        self.active_name_idx = 0
        self.name_edit = ""

    def enroll_setup_screen(self, canvas, key):
        self.draw_header(canvas, "Enrollment Setup", "Pick how many markers, type names, then Start Enrollment.")

        # count control
        draw_rounded_rect(canvas, 60, 130, 420, 220, r=16, color=PANEL, thickness=-1)
        put_text(canvas, "Markers to enroll", (84, 165), scale=0.62, color=TEXT)
        put_text(canvas, f"{self.enroll_count}", (220, 205), scale=1.0, thick=3, color=ACCENT)

        minus = Button("-", (90, 180, 60, 46), enabled=(self.enroll_count > 1), tag="minus")
        plus = Button("+", (330, 180, 60, 46), enabled=(self.enroll_count < MAX_MARKERS), tag="plus")
        for b in (minus, plus):
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b)

        # name entries panel
        draw_rounded_rect(canvas, 470, 130, 1220, 360, r=16, color=PANEL, thickness=-1)
        put_text(canvas, "Marker Names", (494, 165), scale=0.62, color=TEXT)

        # list rows
        rows = []
        for i in range(self.enroll_count):
            yy = 190 + i * 42
            r = (500, yy - 24, 660, 34)  # click to select
            rows.append(r)
            selected = (i == self.active_name_idx)
            col = ACCENT2 if selected else (80, 80, 92)
            draw_rounded_rect(canvas, r[0], r[1], r[0] + r[2], r[1] + r[3], r=12, color=col, thickness=-1)

            name = self.enroll_names[i] if i < len(self.enroll_names) else ""
            if selected:
                shown = self.name_edit if self.name_edit != "" else name
                put_text(canvas, f"Marker {i+1}: {shown}", (520, yy), scale=0.58, color=TEXT)
                put_text(canvas, "Type, then press Enter", (920, yy), scale=0.48, color=MUTED)
            else:
                shown = name if name else f"(click to name)"
                put_text(canvas, f"Marker {i+1}: {shown}", (520, yy), scale=0.58, color=TEXT)

        # bottom controls
        back = Button("Back", (60, 640, 180, 56), tag="back")
        start = Button("Start Enrollment", (980, 640, 240, 56), tag="start", enabled=True)
        for b in (back, start):
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b)

        # handle keyboard for active name
        if key != -1:
            if key in (13, 10):  # Enter commits name
                nm = self.name_edit.strip()
                if nm == "":
                    nm = f"marker{self.active_name_idx+1}"
                while len(self.enroll_names) < self.enroll_count:
                    self.enroll_names.append("")
                self.enroll_names[self.active_name_idx] = nm
                self.name_edit = ""
            else:
                self.handle_keyboard_for_text(key)

        # click actions
        if self.consume_click():
            if minus.enabled and minus.hover:
                self.enroll_count -= 1
                self.active_name_idx = min(self.active_name_idx, self.enroll_count - 1)
            elif plus.enabled and plus.hover:
                self.enroll_count += 1
            elif back.hover:
                self.screen = "main"
            elif start.hover:
                # fill missing names
                while len(self.enroll_names) < self.enroll_count:
                    self.enroll_names.append(f"marker{len(self.enroll_names)+1}")
                for i in range(self.enroll_count):
                    if self.enroll_names[i].strip() == "":
                        self.enroll_names[i] = f"marker{i+1}"
                self.start_enroll_capture()
            else:
                # selecting name row
                for i, r in enumerate(rows):
                    rx, ry, rw, rh = r
                    if point_in_rect(self.mouse_x, self.mouse_y, (rx, ry, rw, rh)):
                        self.active_name_idx = i
                        self.name_edit = ""
                        break

    def start_enroll_capture(self):
        self.screen = "enroll_capture"
        self.enroll_marker_idx = 0
        self.enroll_step_idx = 0
        self.enroll_step_t0 = time.time()
        self.enroll_t0 = time.time()
        self.enroll_embs = []
        self.enroll_frames = 0
        self.enroll_used = 0
        self.registry = Registry()

    def enroll_capture_screen(self, frame_bgr, canvas, s, offx, offy):
        name = self.enroll_names[self.enroll_marker_idx]
        step_text, step_dur = ENROLL_STEPS[self.enroll_step_idx]
        elapsed_step = time.time() - self.enroll_step_t0

        # detect best box (on original frame)
        self.enroll_frames += 1
        H, W = frame_bgr.shape[:2]
        res = self.yolo.predict(frame_bgr, conf=DET_CONF, iou=DET_IOU, verbose=False)

        best_crop_rgb = None
        best_box = None

        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes.data.cpu().numpy()
            boxes = boxes[np.argsort(-boxes[:, 4])]
            x1, y1, x2, y2, conf, cls = boxes[0].tolist()
            bw = x2 - x1
            bh = y2 - y1
            x1p = x1 - BOX_PAD_FRAC * bw
            y1p = y1 - BOX_PAD_FRAC * bh
            x2p = x2 + BOX_PAD_FRAC * bw
            y2p = y2 + BOX_PAD_FRAC * bh
            b = clamp_box(x1p, y1p, x2p, y2p, W, H)
            if b is not None:
                x1i, y1i, x2i, y2i = b
                area = (x2i - x1i) * (y2i - y1i)
                if area >= MIN_BOX_AREA:
                    crop = frame_bgr[y1i:y2i, x1i:x2i]
                    if crop.size > 0 and lap_var(crop) >= BLUR_THRES:
                        best_crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        best_box = (x1i, y1i, x2i, y2i)

        if best_crop_rgb is not None:
            z = embed_crops(self.embedder, [best_crop_rgb], DEVICE)
            if z.shape[0] == 1:
                self.enroll_embs.append(z[0])
                self.enroll_used += 1

        # step advance
        if elapsed_step >= step_dur:
            self.enroll_step_idx += 1
            self.enroll_step_t0 = time.time()

            if self.enroll_step_idx >= len(ENROLL_STEPS):
                # finalize this marker
                embs = np.array(self.enroll_embs, dtype=np.float32)
                prof = build_profile_from_enrollment(name, embs)
                prof.enroll_frames = int(self.enroll_frames)
                prof.enroll_used = int(self.enroll_used)
                self.registry.add_marker(prof)

                # next marker or finish
                self.enroll_marker_idx += 1
                if self.enroll_marker_idx >= len(self.enroll_names):
                    self.save_registry()
                    self.screen = "main"
                    return
                # reset for next marker
                self.enroll_step_idx = 0
                self.enroll_step_t0 = time.time()
                self.enroll_t0 = time.time()
                self.enroll_embs = []
                self.enroll_frames = 0
                self.enroll_used = 0
                return

        # draw overlay
        self.draw_header(canvas, "Enrollment", f"Marker {self.enroll_marker_idx+1}/{len(self.enroll_names)}: {name}")

        # draw step card
        draw_rounded_rect(canvas, 40, 100, 520, 240, r=16, color=PANEL, thickness=-1)
        put_text(canvas, "Instruction", (64, 138), scale=0.62, color=TEXT)
        put_text(canvas, step_text, (64, 178), scale=0.80, thick=2, color=ACCENT2)
        put_text(canvas, f"Used embeddings: {self.enroll_used}", (64, 216), scale=0.55, color=MUTED)

        # progress bar
        total = sum(d for _, d in ENROLL_STEPS)
        elapsed_total = time.time() - self.enroll_t0
        prog = min(1.0, elapsed_total / max(total, 1e-6))
        bar_x1, bar_y1, bar_x2, bar_y2 = 40, 260, 520, 286
        draw_rounded_rect(canvas, bar_x1, bar_y1, bar_x2, bar_y2, r=10, color=(70, 70, 78), thickness=-1)
        fill = int((bar_x2 - bar_x1) * prog)
        draw_rounded_rect(canvas, bar_x1, bar_y1, bar_x1 + fill, bar_y2, r=10, color=ACCENT, thickness=-1)
        put_text(canvas, f"{int(elapsed_total)}/{int(total)}s", (410, 254), scale=0.50, color=TEXT)

        # draw box on canvas if present
        if best_box is not None:
            x1i, y1i, x2i, y2i = best_box
            cx1 = int(offx + x1i * s)
            cy1 = int(offy + y1i * s)
            cx2 = int(offx + x2i * s)
            cy2 = int(offy + y2i * s)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), OK, 2)

        # back button
        back = Button("Cancel", (40, 640, 180, 56), tag="cancel")
        back.hover = point_in_rect(self.mouse_x, self.mouse_y, back.rect)
        draw_button(canvas, back)

        if self.consume_click() and back.hover:
            # abandon enrollment
            self.load_registry_if_any()
            self.screen = "main"

    # ---------------- Trial setup + run ----------------

    def start_trial_setup(self, trial_type: str):
        self.screen = "trial_setup"
        self.trial_type = trial_type
        if trial_type == "swipe":
            self.trial_duration = TRIAL_SWIPE_SEC
        elif trial_type == "interfere":
            self.trial_duration = TRIAL_INTERFERE_SEC
        else:
            self.trial_duration = TRIAL_REENTRY_SEC

    def trial_setup_screen(self, canvas):
        self.draw_header(canvas, "Trial Setup", "Choose condition, then Start. Assign GT by clicking a box then clicking a marker button.")

        draw_rounded_rect(canvas, 60, 120, 600, 320, r=16, color=PANEL, thickness=-1)
        put_text(canvas, f"Trial: {self.trial_type}", (88, 165), scale=0.80, color=ACCENT2)
        put_text(canvas, f"Duration: {self.trial_duration}s", (88, 210), scale=0.62, color=TEXT)

        # toggles
        near_btn = Button("Near", (90, 250, 200, 56), tag="near")
        far_btn = Button("Far", (320, 250, 200, 56), tag="far")
        near_btn.toggled = self.condition_near
        far_btn.toggled = not self.condition_near

        bright_btn = Button("Bright", (90, 320, 200, 56), tag="bright")
        dim_btn = Button("Dim", (320, 320, 200, 56), tag="dim")
        bright_btn.toggled = self.condition_bright
        dim_btn.toggled = not self.condition_bright

        for b in (near_btn, far_btn, bright_btn, dim_btn):
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b)

        start = Button("Start Trial", (980, 640, 240, 56), tag="start")
        back = Button("Back", (60, 640, 180, 56), tag="back")
        for b in (back, start):
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b)

        if self.consume_click():
            if near_btn.hover:
                self.condition_near = True
            elif far_btn.hover:
                self.condition_near = False
            elif bright_btn.hover:
                self.condition_bright = True
            elif dim_btn.hover:
                self.condition_bright = False
            elif back.hover:
                self.screen = "main"
            elif start.hover:
                self.start_trial_run()

    def start_trial_run(self):
        self.screen = "trial_run"
        self.trial_t0 = time.time()
        self.trial_id = f"{now_ms()}_{self.trial_type}"
        self.frame_idx = 0
        self.states = {}
        self.active_prev = set()
        self.selected_track = None
        self.gt_map = {}
        self.frame_cols, self.frame_rows = new_frame_logger()
        self.event_cols, self.event_rows = new_event_logger()
        self.lat_ms = []

    def trial_run_screen(self, frame_bgr, canvas, s, offx, offy):
        assert self.registry is not None

        t_frame0 = time.time()
        self.frame_idx += 1

        # YOLO + ByteTrack
        t_det0 = time.time()
        r = self.yolo.track(
            frame_bgr,
            tracker=TRACKER_YAML,
            persist=True,
            conf=DET_CONF,
            iou=DET_IOU,
            verbose=False
        )[0]
        det_track_ms = (time.time() - t_det0) * 1000.0

        boxes = r.boxes
        det_meta = []  # (tid, (x1,y1,x2,y2), conf)
        crops_rgb = []

        H, W = frame_bgr.shape[:2]
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(boxes),), dtype=np.float32)
            tids = None
            if hasattr(boxes, "id") and boxes.id is not None:
                tids = boxes.id.cpu().numpy().astype(int)
            else:
                tids = -np.ones((len(boxes),), dtype=np.int32)

            # sort by confidence
            order = np.argsort(-confs)
            for i in order:
                tid = int(tids[i])
                x1, y1, x2, y2 = xyxy[i].tolist()
                c = float(confs[i])

                bw = x2 - x1
                bh = y2 - y1
                x1p = x1 - BOX_PAD_FRAC * bw
                y1p = y1 - BOX_PAD_FRAC * bh
                x2p = x2 + BOX_PAD_FRAC * bw
                y2p = y2 + BOX_PAD_FRAC * bh

                b = clamp_box(x1p, y1p, x2p, y2p, W, H)
                if b is None:
                    continue
                x1i, y1i, x2i, y2i = b

                area = (x2i - x1i) * (y2i - y1i)
                if area < MIN_BOX_AREA:
                    continue

                crop = frame_bgr[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    continue
                if lap_var(crop) < BLUR_THRES:
                    continue

                det_meta.append((tid, (x1i, y1i, x2i, y2i), c))
                crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # embed
        t_emb0 = time.time()
        Z = embed_crops(self.embedder, crops_rgb, DEVICE)
        embed_ms = (time.time() - t_emb0) * 1000.0

        # match (per track with EMA)
        t_match0 = time.time()
        preds = []  # (tid, pred, sim, box)
        active_now = set()

        for i in range(Z.shape[0]):
            tid, box, conf = det_meta[i]
            z = Z[i]

            if tid < 0:
                pred, sim = match_marker(z, self.registry)
                preds.append((tid, pred, sim, box))
                continue

            active_now.add(tid)
            if tid not in self.states:
                self.states[tid] = TrackState(track_id=tid, last_seen_frame=self.frame_idx)
                self.event_rows.append([self.trial_id, self.trial_type, self.condition_str(), now_ms(), "track_new", tid, "", "", ""])
            self.states[tid].last_seen_frame = self.frame_idx

            pred, sim = update_identity(self.states[tid], z, self.registry)
            preds.append((tid, pred, float(sim), box))

        match_ms = (time.time() - t_match0) * 1000.0

        new_tracks = len(active_now - self.active_prev)
        lost_tracks = len(self.active_prev - active_now)
        if lost_tracks > 0:
            for tid_lost in (self.active_prev - active_now):
                self.event_rows.append([self.trial_id, self.trial_type, self.condition_str(), now_ms(), "track_lost", tid_lost, "", "", ""])
        self.active_prev = active_now

        # unknown tracks count
        unknown_tracks = sum(1 for (tid, pred, sim, box) in preds if pred == "unknown")

        # accuracy if GT assigned
        classes = self.registry.names()
        c2i = {c: i for i, c in enumerate(classes)}
        if not hasattr(self, "_cm"):
            self._cm = np.zeros((len(classes), len(classes)), dtype=np.int32)
            self._gt_total = 0
            self._gt_correct = 0

        for (tid, pred, sim, box) in preds:
            if tid in self.gt_map and pred in c2i and self.gt_map[tid] in c2i:
                gt = self.gt_map[tid]
                self._cm[c2i[gt], c2i[pred]] += 1
                self._gt_total += 1
                if gt == pred:
                    self._gt_correct += 1

        # clickable boxes (canvas coords)
        clickable_boxes = []  # (tid, cx1,cy1,cx2,cy2)
        for (tid, pred, sim, box) in preds:
            x1i, y1i, x2i, y2i = box
            cx1 = int(offx + x1i * s)
            cy1 = int(offy + y1i * s)
            cx2 = int(offx + x2i * s)
            cy2 = int(offy + y2i * s)
            clickable_boxes.append((tid, cx1, cy1, cx2, cy2))

            col = OK if pred != "unknown" else UNKNOWN
            if self.selected_track is not None and tid == self.selected_track:
                col = (255, 255, 255)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), col, 2)

            label = f"T{tid} {pred} {sim:.2f}"
            if tid in self.gt_map:
                label += f" | GT:{self.gt_map[tid]}"
            put_text(canvas, label, (cx1, max(18, cy1 - 8)), scale=0.50, thick=2, color=TEXT)

        # GT assignment panel (click marker buttons)
        draw_rounded_rect(canvas, 20, 100, 340, 690, r=16, color=PANEL, thickness=-1)
        put_text(canvas, "GT Assignment", (42, 140), scale=0.62, color=TEXT)
        put_text(canvas, "Click a box, then click a name", (42, 170), scale=0.48, color=MUTED)

        gt_toggle = Button("GT: ON" if self.gt_enabled else "GT: OFF", (42, 190, 260, 50), tag="gt")
        gt_toggle.toggled = self.gt_enabled
        gt_toggle.hover = point_in_rect(self.mouse_x, self.mouse_y, gt_toggle.rect)
        draw_button(canvas, gt_toggle)

        y = 260
        marker_buttons = []
        for nm in classes:
            b = Button(nm, (42, y, 260, 52), tag=f"gt_{nm}", enabled=self.gt_enabled and (self.selected_track is not None))
            b.hover = point_in_rect(self.mouse_x, self.mouse_y, b.rect)
            marker_buttons.append(b)
            draw_button(canvas, b)
            y += 62

        clear_sel = Button("Clear Selection", (42, 610, 260, 50), enabled=(self.selected_track is not None), tag="clear")
        clear_sel.hover = point_in_rect(self.mouse_x, self.mouse_y, clear_sel.rect)
        draw_button(canvas, clear_sel)

        # HUD
        remaining = max(0.0, self.trial_duration - (time.time() - self.trial_t0))
        acc = (self._gt_correct / self._gt_total) if self._gt_total > 0 else None

        self.draw_header(
            canvas,
            f"Trial: {self.trial_type} | Condition: {self.condition_str()} | Remaining: {remaining:.1f}s",
            f"Dets:{len(preds)} Tracks:{len(active_now)}  Lat(ms): det+track {det_track_ms:.1f} embed {embed_ms:.1f} match {match_ms:.1f}"
        )
        if acc is not None:
            put_text(canvas, f"GT accuracy: {acc*100:.1f}%  (samples={self._gt_total})", (380, 110), scale=0.58, color=OK)

        # stop button
        stop = Button("End Trial", (1040, 640, 220, 56), tag="stop")
        stop.hover = point_in_rect(self.mouse_x, self.mouse_y, stop.rect)
        draw_button(canvas, stop)

        # clicks
        if self.consume_click():
            # select track by clicking a box
            for (tid, cx1, cy1, cx2, cy2) in clickable_boxes:
                if point_in_rect(self.mouse_x, self.mouse_y, (cx1, cy1, cx2 - cx1, cy2 - cy1)):
                    self.selected_track = tid
                    break

            # gt toggle
            if gt_toggle.hover:
                self.gt_enabled = not self.gt_enabled

            # assign gt by clicking marker name
            for b in marker_buttons:
                if b.enabled and b.hover and b.tag.startswith("gt_"):
                    nm = b.tag.replace("gt_", "")
                    self.gt_map[self.selected_track] = nm
                    self.event_rows.append([self.trial_id, self.trial_type, self.condition_str(), now_ms(), "gt_assign", self.selected_track, "", nm, ""])
                    break

            if clear_sel.enabled and clear_sel.hover:
                self.selected_track = None

            if stop.hover:
                self.finish_trial()
                return

        # timing and logs
        total_ms = (time.time() - t_frame0) * 1000.0
        self.lat_ms.append(total_ms)

        self.frame_rows.append([
            self.trial_id, self.trial_type, self.condition_str(),
            self.frame_idx, now_ms(),
            len(preds), len(active_now),
            round(det_track_ms, 3), round(embed_ms, 3), round(match_ms, 3), round(total_ms, 3),
            new_tracks, lost_tracks,
            unknown_tracks
        ])

        # auto-end
        if remaining <= 0.0:
            self.finish_trial()

    def finish_trial(self):
        ensure_dir(RUN_DIR)
        out_trial_dir = os.path.join(RUN_DIR, "trials", self.trial_id)
        ensure_dir(out_trial_dir)

        frame_csv = os.path.join(out_trial_dir, "frame_log.csv")
        pd.DataFrame(self.frame_rows, columns=self.frame_cols).to_csv(frame_csv, index=False)

        event_csv = os.path.join(out_trial_dir, "event_log.csv")
        pd.DataFrame(self.event_rows, columns=self.event_cols).to_csv(event_csv, index=False)

        lat = np.array(self.lat_ms, dtype=np.float32)
        summary = {
            "trial_id": self.trial_id,
            "trial_type": self.trial_type,
            "condition": self.condition_str(),
            "frames": int(self.frame_idx),
            "duration_sec": float(self.trial_duration),
            "mean_total_latency_ms": float(lat.mean()) if lat.size else None,
            "p95_total_latency_ms": float(np.percentile(lat, 95)) if lat.size else None,
            "mean_fps": float(self.frame_idx / max(self.trial_duration, 1e-6)),
            "track_count": int(len(self.states)),
            "avg_switches_per_track": float(np.mean([s.switches for s in self.states.values()])) if len(self.states) else 0.0,
            "frame_log_csv": frame_csv,
            "event_log_csv": event_csv,
            "gt_samples": int(getattr(self, "_gt_total", 0)),
            "top1_accuracy": (float(self._gt_correct / self._gt_total) if getattr(self, "_gt_total", 0) > 0 else None),
        }

        # confusion matrix only if gt exists
        if getattr(self, "_gt_total", 0) > 0 and self.registry is not None:
            cm_path = os.path.join(out_trial_dir, "confusion_matrix.png")
            save_confusion_matrix(self.registry.names(), self._cm, cm_path)
            summary["confusion_matrix_png"] = cm_path
            summary["confusion_matrix"] = self._cm.tolist()
            summary["classes"] = self.registry.names()

        summ_path = os.path.join(out_trial_dir, "trial_summary.json")
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # reset GT accumulators for next run
        if hasattr(self, "_cm"):
            delattr(self, "_cm")
        if hasattr(self, "_gt_total"):
            delattr(self, "_gt_total")
        if hasattr(self, "_gt_correct"):
            delattr(self, "_gt_correct")

        self.screen = "main"

    # ---------------- main tick ----------------

    def tick(self, frame_bgr, key):
        canvas, s, offx, offy = compose_canvas(frame_bgr)

        if self.screen == "main":
            self.main_screen(canvas)

        elif self.screen == "enroll_setup":
            self.enroll_setup_screen(canvas, key)

        elif self.screen == "enroll_capture":
            self.enroll_capture_screen(frame_bgr, canvas, s, offx, offy)

        elif self.screen == "trial_setup":
            self.trial_setup_screen(canvas)

        elif self.screen == "trial_run":
            self.trial_run_screen(frame_bgr, canvas, s, offx, offy)

        cv2.imshow("PINCH Live", canvas)


def main():
    set_seed(0)
    ensure_dir(RUN_DIR)

    if not os.path.isfile(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO_WEIGHTS not found: {YOLO_WEIGHTS}")
    if not os.path.isfile(EMBEDDER_WEIGHTS):
        raise FileNotFoundError(f"EMBEDDER_WEIGHTS not found: {EMBEDDER_WEIGHTS}")

    cap = open_source()
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera/video source.")

    yolo = YOLO(YOLO_WEIGHTS)
    embedder = load_embedder(EMBEDDER_WEIGHTS, DEVICE)

    app = PINCHApp(yolo, embedder)

    while app.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC hard exit
            break

        app.tick(frame, key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()