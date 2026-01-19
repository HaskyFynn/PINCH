"""
PINCH Live System (single script)
- Modern OpenCV UI
- Live mode (webcam): Enrollment + Trials
- Demo mode (video): Demo Enrollment + Demo Trials
- Pipeline: YOLO + ByteTrack + ResNet embedder + prototype matching
- Logging: minimal, paper-relevant CSV + JSON summary + confusion matrix (optional GT)

Install:
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow

Notes:
- Windows file picker uses a Tk fallback and Win32 dialog. If it fails, paste a path (Ctrl+V) in the UI field.
- If MOV decode fails, convert to MP4 (H.264) then retry.
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
# USER SETTINGS
# ============================================================
YOLO_WEIGHTS = r"C:\Users\USER\Desktop\my_pinch\Pinch-Model\hasky\letsgo\best.pt"
EMBEDDER_WEIGHTS = r"C:\Users\USER\Desktop\my_pinch\Pinch-Model\hasky\letsgo\embedder_resnet18_triplet.pt"
RUN_DIR = r"C:\Users\USER\Desktop\my_pinch\Pinch-Model\hasky\letsgo\pinch_live"

WEBCAM_INDEX = 0

# Fixed canvas for consistent UI
CANVAS_W = 1280
CANVAS_H = 720

# Detection + tracking
DET_CONF = 0.15
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
THRESH_PERCENTILE = 5

# Track smoothing
EMA_ALPHA = 0.70
DECISION_HYST = 0.05

# Enrollment steps (seconds)
ENROLL_STEPS = [
    ("Hold front-facing", 8),
    ("Swipe left-right", 10),
    ("Swipe up-down", 10),
    ("Hold left side", 6),
    ("Hold right side", 6),
    ("Tilt + motion", 8),
]

# Trial durations
TRIAL_SWIPE_SEC = 12
TRIAL_INTERFERE_SEC = 18
TRIAL_REENTRY_SEC = 18

# ============================================================
# UI THEME (studio)
# ============================================================
BG = (18, 22, 32)
BG_TOP = (10, 14, 22)
BG_GLOW_1 = (28, 62, 90)
BG_GLOW_2 = (20, 80, 120)
CARD = (26, 30, 40)
CARD2 = (32, 38, 52)
STROKE = (70, 82, 100)
ACCENT = (64, 200, 255)
ACCENT_HI = (90, 224, 255)
ACCENT_SOFT = (80, 150, 210)
OK = (72, 220, 150)
WARN = (80, 95, 255)
TEXT = (240, 242, 248)
MUTED = (170, 178, 188)
WHITE = (250, 250, 252)
BTN_SECONDARY = (84, 94, 116)
BTN_DISABLED = (60, 66, 78)
INPUT_BG = (22, 26, 36)
INPUT_ACTIVE = (30, 38, 58)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX
UI_SCALE = 1.08

TOPBAR_H = 76
SIDEBAR_W = 360
UI_PAD = 18

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".m4v", ".MP4", ".AVI", ".MOV", ".M4V")

# ============================================================
# HELPERS
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

def draw_text(img, s, x, y, scale=0.6, color=TEXT, thick=2, font=None, shadow=False):
    f = font or FONT
    scale = scale * UI_SCALE
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), f, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), f, scale, color, thick, cv2.LINE_AA)

def text_size(s, scale=0.6, thick=2, font=None):
    f = font or FONT
    return cv2.getTextSize(s, f, scale * UI_SCALE, thick)[0]

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def rounded_rect(img, x1, y1, x2, y2, r=14, color=CARD, thickness=-1):
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

def shadow_card(img, x1, y1, x2, y2, r=16, shadow=10):
    alpha_rect(img, x1 + shadow, y1 + shadow, x2 + shadow, y2 + shadow, (0, 0, 0), alpha=0.28)
    rounded_rect(img, x1, y1, x2, y2, r=r, color=CARD, thickness=-1)
    rounded_rect(img, x1, y1, x2, y2, r=r, color=STROKE, thickness=2)
    cv2.line(img, (x1 + r, y1 + 2), (x2 - r, y1 + 2), tint(CARD, 0.18), 1)

_BG_CACHE = None
_BG_CACHE_SHAPE = None

def gradient_bg(canvas):
    global _BG_CACHE, _BG_CACHE_SHAPE
    h, w = canvas.shape[:2]
    if _BG_CACHE is None or _BG_CACHE_SHAPE != (h, w):
        top = np.array(BG_TOP, dtype=np.float32)
        bot = np.array(BG, dtype=np.float32)
        grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        col = (1 - grad) * top + grad * bot
        bg = np.repeat(col[:, None, :], w, axis=1).astype(np.uint8)

        glow = bg.copy()
        cv2.circle(glow, (int(w * 0.18), int(h * 0.08)), int(h * 0.65), BG_GLOW_1, -1)
        cv2.circle(glow, (int(w * 0.88), int(h * 0.12)), int(h * 0.55), BG_GLOW_2, -1)
        cv2.addWeighted(glow, 0.18, bg, 0.82, 0, bg)

        noise = np.random.randint(0, 6, size=(h, w, 1), dtype=np.uint8)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

        _BG_CACHE = bg
        _BG_CACHE_SHAPE = (h, w)
    canvas[:] = _BG_CACHE

def point_in(x, y, r):
    rx, ry, rw, rh = r
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def truncate_path(p, max_chars=52):
    if not p:
        return ""
    if len(p) <= max_chars:
        return p
    head = p[: int(max_chars * 0.35)]
    tail = p[-int(max_chars * 0.55):]
    return head + "..." + tail

def window_closed(win_name: str) -> bool:
    try:
        v = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        return v < 1
    except Exception:
        return True

def mix_color(a, b, t: float) -> Tuple[int, int, int]:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))

def tint(color, t: float) -> Tuple[int, int, int]:
    return mix_color(color, (255, 255, 255), t)

def shade(color, t: float) -> Tuple[int, int, int]:
    return mix_color(color, (0, 0, 0), t)

def normalize_path(p: str) -> str:
    if not p:
        return ""
    p = p.strip().strip('"').strip("'")
    if p.lower().startswith("file://"):
        p = p.replace("file:///", "", 1)
        p = p.replace("file://", "", 1)
    p = os.path.expandvars(os.path.expanduser(p))
    if os.name == "nt":
        p = p.replace("/", "\\")
    return os.path.normpath(p)

def get_clipboard_text() -> str:
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        CF_UNICODETEXT = 13
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(None):
            return ""
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            user32.CloseClipboard()
            return ""
        pcontents = kernel32.GlobalLock(handle)
        if not pcontents:
            user32.CloseClipboard()
            return ""
        data = ctypes.wstring_at(pcontents)
        kernel32.GlobalUnlock(handle)
        user32.CloseClipboard()
        return data
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.CloseClipboard()
        except Exception:
            pass
        return ""

def sanitize_clipboard_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n").split("\n")[0]
    return text.strip()

def pick_file_tk(title: str, filetypes: List[Tuple[str, str]]) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path
    except Exception:
        return ""

# ============================================================
# WINDOWS FILE PICKER (tk fallback + ctypes)
# ============================================================
def pick_file_win32(title="Select file", filter_str="All Files\0*.*\0\0") -> Tuple[str, int]:
    try:
        import ctypes
        from ctypes import wintypes

        # Flags
        OFN_EXPLORER = 0x00080000
        OFN_FILEMUSTEXIST = 0x00001000
        OFN_PATHMUSTEXIST = 0x00000800
        OFN_NOCHANGEDIR = 0x00000008
        OFN_DONTADDTORECENT = 0x02000000

        class OPENFILENAMEW(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", wintypes.LPVOID),
                ("lpTemplateName", wintypes.LPCWSTR),
                ("pvReserved", wintypes.LPVOID),
                ("dwReserved", wintypes.DWORD),
                ("FlagsEx", wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        try:
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
        except Exception:
            pass

        buf = ctypes.create_unicode_buffer(4096)
        buf.value = ""

        ofn = OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
        ofn.hwndOwner = hwnd
        ofn.lpstrFilter = filter_str
        ofn.lpstrFile = buf
        ofn.nMaxFile = 4096
        ofn.lpstrTitle = title
        ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_DONTADDTORECENT

        ok = ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn))
        if ok:
            return buf.value, 0
        err = ctypes.windll.comdlg32.CommDlgExtendedError()
        return "", int(err)
    except Exception:
        return "", 1

def pick_file_windows(title="Select file", filter_str="All Files\0*.*\0\0", filetypes: Optional[List[Tuple[str, str]]] = None) -> str:
    if os.name != "nt":
        return ""

    if filetypes:
        path = pick_file_tk(title, filetypes)
        if path:
            return path

    path, err = pick_file_win32(title=title, filter_str=filter_str)
    if path:
        return path

    if err != 0:
        fallback_types = filetypes or [("All Files", "*.*")]
        return pick_file_tk(title, fallback_types)
    return ""

def pick_video_file_windows(title="Select video") -> str:
    flt = "Video Files\0*.mp4;*.avi;*.mov;*.m4v\0All Files\0*.*\0\0"
    ftypes = [("Video Files", "*.mp4 *.avi *.mov *.m4v"), ("All Files", "*.*")]
    return pick_file_windows(title=title, filter_str=flt, filetypes=ftypes)

def pick_registry_file_windows(title="Select registry json") -> str:
    flt = "JSON\0*.json\0All Files\0*.*\0\0"
    ftypes = [("JSON", "*.json"), ("All Files", "*.*")]
    return pick_file_windows(title=title, filter_str=flt, filetypes=ftypes)

# ============================================================
# VIDEO SOURCE
# ============================================================
class VideoSource:
    def __init__(self):
        self.cap = None
        self.mode = "webcam"  # webcam or video
        self.webcam_index = WEBCAM_INDEX
        self.video_path = ""
        self.loop = True

    def open_webcam(self, index=0):
        self.release()
        self.mode = "webcam"
        self.webcam_index = index
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        self.cap = cap
        return self.is_opened()

    def open_video(self, path: str, loop=True):
        self.release()
        self.mode = "video"
        path = normalize_path(path)
        if path:
            path = os.path.abspath(path)
        self.video_path = path
        self.loop = loop
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        self.cap = cap
        return self.is_opened()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.is_opened():
            return False, None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return True, frame

        if self.mode == "video" and self.loop and self.video_path:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok2, frame2 = self.cap.read()
                if ok2 and frame2 is not None:
                    return True, frame2
            except Exception:
                pass
        return False, None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def describe(self):
        if self.mode == "webcam":
            return f"Webcam {self.webcam_index}"
        if self.video_path:
            return "Video: " + os.path.basename(self.video_path)
        return "Video"

# ============================================================
# CANVAS COMPOSITION
# ============================================================
def compose_canvas(frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int, Tuple[int, int, int, int]]:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    gradient_bg(canvas)

    left_w = SIDEBAR_W
    pad = UI_PAD
    vx1, vy1 = left_w + pad, TOPBAR_H + 16
    vx2, vy2 = CANVAS_W - pad, CANVAS_H - pad
    shadow_card(canvas, vx1, vy1, vx2, vy2, r=18, shadow=10)

    fh, fw = frame_bgr.shape[:2]
    vw, vh = (vx2 - vx1) - 16, (vy2 - vy1) - 16
    s = min(vw / max(fw, 1), vh / max(fh, 1))
    s = max(min(s, 1.0), 1e-6)
    rw, rh = int(fw * s), int(fh * s)
    resized = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)

    offx = vx1 + 8 + (vw - rw) // 2
    offy = vy1 + 8 + (vh - rh) // 2
    canvas[offy:offy + rh, offx:offx + rw] = resized

    view_rect = (offx, offy, rw, rh)
    return canvas, s, offx, offy, view_rect

# ============================================================
# EMBEDDER
# ============================================================
class ResnetEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = m.fc.in_features
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
# REGISTRY + PROTOTYPES
# ============================================================
@dataclass
class MarkerProfile:
    marker_id: str
    proto: List[List[float]]
    thr: float
    enroll_frames: int
    enroll_used: int
    source_mode: str = ""
    source_path: str = ""

class Registry:
    def __init__(self):
        self.markers: List[MarkerProfile] = []

    def add_marker(self, profile: MarkerProfile):
        # replace same name if exists
        self.markers = [m for m in self.markers if m.marker_id != profile.marker_id]
        self.markers.append(profile)

    def names(self) -> List[str]:
        return [m.marker_id for m in self.markers]

    def to_json(self) -> Dict:
        return {"markers": [asdict(m) for m in self.markers]}

    @staticmethod
    def from_json(d: Dict) -> "Registry":
        r = Registry()
        for m in d.get("markers", []):
            r.markers.append(MarkerProfile(**m))
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

def build_profile_from_enrollment(marker_id: str, embs: np.ndarray) -> Tuple[List[List[float]], float]:
    if embs.shape[0] < 10:
        protos = kmeans_prototypes(embs, k=min(2, embs.shape[0]), iters=8, seed=0)
    else:
        protos = kmeans_prototypes(embs, k=PROTOS_PER_MARKER, iters=PROTOS_KMEANS_ITERS, seed=0)

    sims = embs @ protos.T if protos.shape[0] > 0 else np.zeros((embs.shape[0], 1), dtype=np.float32)
    best = sims.max(axis=1) if sims.shape[1] > 0 else np.zeros((embs.shape[0],), dtype=np.float32)
    thr = float(np.percentile(best, THRESH_PERCENTILE)) if best.shape[0] > 0 else 0.0
    return protos.tolist(), thr

def match_marker(emb: np.ndarray, registry: Registry) -> Tuple[str, float]:
    best_name = "unknown"
    best_sim = -1.0
    best_thr = None
    for m in registry.markers:
        P = np.array(m.proto, dtype=np.float32)
        if P.size == 0:
            continue
        sim = float(np.max(P @ emb))
        if sim > best_sim:
            best_sim = sim
            best_name = m.marker_id
            best_thr = m.thr
    if best_name != "unknown" and best_thr is not None and best_sim < best_thr:
        return "unknown", best_sim
    return best_name, best_sim

# ============================================================
# TRACK STATE
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
    if ts.z_ema is None:
        ts.z_ema = z.copy()
    else:
        ts.z_ema = safe_norm(EMA_ALPHA * ts.z_ema + (1.0 - EMA_ALPHA) * z)

    pred, sim = match_marker(ts.z_ema, registry)

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
# LOGGING (paper relevant)
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
# UI PRIMITIVES
# ============================================================
@dataclass
class Button:
    text: str
    rect: Tuple[int, int, int, int]
    enabled: bool = True
    hover: bool = False
    toggled: bool = False
    tag: str = ""

def draw_button(canvas, b: Button, primary=True):
    x, y, w, h = b.rect
    base = ACCENT if primary else BTN_SECONDARY

    if not b.enabled:
        col = BTN_DISABLED
        stroke = shade(col, 0.25)
        txt = MUTED
    else:
        if b.toggled:
            col = mix_color(ACCENT, OK, 0.35)
        else:
            col = base
        if b.hover:
            col = tint(col, 0.12)
        stroke = tint(col, 0.18)
        txt = WHITE

    alpha_rect(canvas, x + 6, y + 8, x + w + 6, y + h + 8, (0, 0, 0), alpha=0.28)
    rounded_rect(canvas, x, y, x + w, y + h, r=16, color=col, thickness=-1)
    alpha_rect(canvas, x + 2, y + 2, x + w - 2, y + h // 2, tint(col, 0.2), alpha=0.25)
    rounded_rect(canvas, x, y, x + w, y + h, r=16, color=stroke, thickness=2)

    ts = text_size(b.text, scale=0.6, thick=2)
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2
    draw_text(canvas, b.text, tx, ty, scale=0.6, color=txt, thick=2)

def draw_input(canvas, rect, label, value, placeholder="", active=False, hint=""):
    x, y, w, h = rect
    draw_text(canvas, label, x, y - 10, scale=0.48, color=MUTED, thick=2)
    bg = INPUT_ACTIVE if active else INPUT_BG
    border = ACCENT if active else STROKE
    rounded_rect(canvas, x, y, x + w, y + h, r=12, color=bg, thickness=-1)
    rounded_rect(canvas, x, y, x + w, y + h, r=12, color=border, thickness=2)

    display = value if value else placeholder
    txt_col = TEXT if value else MUTED
    ts = text_size(display, scale=0.54, thick=2)
    ty = y + (h + ts[1]) // 2
    draw_text(canvas, display, x + 12, ty, scale=0.54, color=txt_col, thick=2)

    if active and int(time.time() * 2) % 2 == 0:
        caret_x = min(x + w - 12, x + 12 + ts[0] + 2)
        cv2.line(canvas, (caret_x, y + 10), (caret_x, y + h - 10), ACCENT_HI, 1)

    if hint:
        draw_text(canvas, hint, x, y + h + 18, scale=0.46, color=MUTED, thick=2)

def pill(canvas, x, y, text, color, text_color=WHITE):
    ts = text_size(text, scale=0.5, thick=2)
    w = ts[0] + 18
    h = 28
    base = tint(color, 0.08)
    rounded_rect(canvas, x, y, x + w, y + h, r=14, color=base, thickness=-1)
    rounded_rect(canvas, x, y, x + w, y + h, r=14, color=shade(base, 0.2), thickness=2)
    draw_text(canvas, text, x + 9, y + 20, scale=0.5, color=text_color, thick=2)
    return x + w + 8

# ============================================================
# APP
# ============================================================
class PINCHApp:
    def __init__(self, yolo: YOLO, embedder: nn.Module):
        self.win = "PINCH Live"
        self.yolo = yolo
        self.embedder = embedder

        self.running = True
        self.screen = "main"  # main, live_enroll, live_trial_setup, live_trial_run, demo_menu, demo_enroll_setup, demo_enroll_run, demo_trial_setup, demo_trial_run

        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked = False
        self.focus_field = ""

        # global video source (switch per mode)
        self.source = VideoSource()
        if not self.source.open_webcam(WEBCAM_INDEX):
            raise RuntimeError("Failed to open webcam.")

        # registry
        self.registry: Optional[Registry] = None
        self.reg_path = os.path.join(RUN_DIR, "registry.json")
        self.load_registry()

        # live enrollment (webcam only)
        self.enroll_name = ""
        self.enroll_name_edit = ""
        self.enroll_step_idx = 0
        self.enroll_step_t0 = 0.0
        self.enroll_t0 = 0.0
        self.enroll_embs: List[np.ndarray] = []
        self.enroll_frames = 0
        self.enroll_used = 0

        # demo enrollment setup
        self.demo_enroll_name = ""
        self.demo_enroll_name_edit = ""
        self.demo_enroll_video = ""
        self.demo_enroll_video_edit = ""

        # demo trial setup
        self.demo_trial_video = ""
        self.demo_trial_video_edit = ""
        self.demo_trial_type = "swipe"
        self.demo_condition_near = True
        self.demo_condition_bright = True
        self.demo_gt_enabled = False

        # trial runtime
        self.trial_type = "swipe"
        self.condition_near = True
        self.condition_bright = True
        self.trial_duration = TRIAL_SWIPE_SEC

        self.trial_id = ""
        self.trial_t0 = 0.0
        self.frame_idx = 0
        self.states: Dict[int, TrackState] = {}
        self.active_prev = set()

        self.selected_track: Optional[int] = None
        self.gt_map: Dict[int, str] = {}
        self.gt_enabled = True

        self.frame_cols, self.frame_rows = new_frame_logger()
        self.event_cols, self.event_rows = new_event_logger()
        self.lat_ms: List[float] = []
        self._cm = None
        self._gt_total = 0
        self._gt_correct = 0

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True

    def consume_click(self) -> bool:
        if self.clicked:
            self.clicked = False
            return True
        return False

    def request_exit(self):
        self.running = False

    def load_registry(self):
        self.registry = None
        if os.path.isfile(self.reg_path):
            try:
                with open(self.reg_path, "r", encoding="utf-8") as f:
                    r = Registry.from_json(json.load(f))
                if len(r.markers) > 0:
                    self.registry = r
            except Exception:
                self.registry = None

    def save_registry(self):
        ensure_dir(RUN_DIR)
        if self.registry is None:
            return
        with open(self.reg_path, "w", encoding="utf-8") as f:
            json.dump(self.registry.to_json(), f, indent=2)

    def condition_str(self) -> str:
        d = "near" if self.condition_near else "far"
        l = "bright" if self.condition_bright else "dim"
        return f"{d}/{l}"

    def draw_topbar(self, canvas, title, subtitle=""):
        alpha_rect(canvas, 0, 0, CANVAS_W, TOPBAR_H, (0, 0, 0), alpha=0.35)
        cv2.line(canvas, (0, TOPBAR_H), (CANVAS_W, TOPBAR_H), STROKE, 1)
        cv2.line(canvas, (18, 14), (18, TOPBAR_H - 14), ACCENT, 3)
        draw_text(canvas, title, 34, 36, scale=0.92, color=ACCENT_HI, thick=2, font=FONT_TITLE)
        if subtitle:
            draw_text(canvas, subtitle, 34, 62, scale=0.5, color=MUTED, thick=2)

        x = CANVAS_W - 430
        x = pill(canvas, x, 22, self.source.describe(), CARD2)
        if self.registry is None:
            pill(canvas, x, 22, "Registry: none", (70, 40, 40))
        else:
            pill(canvas, x, 22, f"Registry: {len(self.registry.markers)}", (30, 60, 30))

    def draw_sidebar_card(self, canvas, title, lines: List[str], y, h):
        x1, y1, x2, y2 = 18, y, 18 + 324, y + h
        shadow_card(canvas, x1, y1, x2, y2, r=16, shadow=8)
        draw_text(canvas, title, x1 + 14, y1 + 28, scale=0.58, color=ACCENT_HI, thick=2, font=FONT_TITLE)
        yy = y1 + 54
        for s in lines:
            draw_text(canvas, s, x1 + 14, yy, scale=0.52, color=MUTED, thick=2)
            yy += 22

    def handle_text_input(self, key, buf: str, max_len=80) -> str:
        if key in (8, 127):
            return buf[:-1]
        if key in (13, 10):
            return buf
        if key == 22:  # Ctrl+V
            clip = sanitize_clipboard_text(get_clipboard_text())
            if clip:
                clip = clip[: max_len - len(buf)]
                return buf + clip
        if 32 <= key <= 126 and len(buf) < max_len:
            return buf + chr(key)
        return buf

    # ---------------- Screens ----------------
    def main_screen(self, canvas) -> None:
        self.draw_topbar(canvas, "PINCHReader Live", "Live uses webcam. Demo uses video files.")

        reg_line = "Registry not found. Run enrollment first." if self.registry is None else "Markers: " + ", ".join(self.registry.names())
        self.draw_sidebar_card(canvas, "Status", [reg_line], y=92, h=90)

        btns = []
        bx, by, bw, bh, gap = 28, 205, 300, 54, 12

        def add(text, tag, enabled=True):
            b = Button(text=text, rect=(bx, by + len(btns) * (bh + gap), bw, bh), enabled=enabled, tag=tag)
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            btns.append(b)

        add("Live Enrollment", "live_enroll", enabled=True)
        add("Live Trials", "live_trials", enabled=(self.registry is not None))
        add("Demo Mode", "demo", enabled=True)
        add("Reload registry", "reload", enabled=True)
        add("Exit", "exit", enabled=True)

        for b in btns:
            primary = b.tag in ("live_enroll", "live_trials", "demo")
            draw_button(canvas, b, primary=primary)

        clicked = self.consume_click()
        if clicked:
            for b in btns:
                if b.enabled and b.hover:
                    if b.tag == "live_enroll":
                        self.start_live_enroll()
                    elif b.tag == "live_trials":
                        self.start_live_trial_setup()
                    elif b.tag == "demo":
                        self.screen = "demo_menu"
                    elif b.tag == "reload":
                        self.load_registry()
                    elif b.tag == "exit":
                        self.request_exit()
                    break

    # ---------------- LIVE ENROLLMENT (webcam) ----------------
    def start_live_enroll(self):
        self.screen = "live_enroll"
        self.focus_field = "live_enroll_name"
        self.enroll_name = ""
        self.enroll_name_edit = ""
        self.enroll_step_idx = 0
        self.enroll_step_t0 = time.time()
        self.enroll_t0 = time.time()
        self.enroll_embs = []
        self.enroll_frames = 0
        self.enroll_used = 0
        # force webcam
        self.source.open_webcam(self.source.webcam_index)

    def live_enroll_screen(self, frame_bgr, canvas, s, offx, offy, key):
        self.draw_topbar(canvas, "Live Enrollment", "Name the marker, press Enter, then follow the prompts.")

        # name card
        shadow_card(canvas, 18, 92, 18 + 324, 92 + 150, r=16, shadow=8)
        name_value = self.enroll_name_edit if self.enroll_name == "" else self.enroll_name
        name_rect = (34, 140, 300, 46)
        draw_input(
            canvas,
            name_rect,
            "Marker name",
            name_value,
            placeholder="Type a name and press Enter",
            active=(self.focus_field == "live_enroll_name" and self.enroll_name == ""),
            hint="Enter locks the name for this run.",
        )

        # handle typing until name confirmed
        if self.enroll_name == "" and key != -1 and self.focus_field == "live_enroll_name":
            if key in (13, 10):
                nm = self.enroll_name_edit.strip()
                self.enroll_name = nm if nm else f"marker_{now_ms()}"
                self.enroll_name_edit = ""
                self.enroll_step_idx = 0
                self.enroll_step_t0 = time.time()
                self.enroll_t0 = time.time()
                self.enroll_embs = []
                self.enroll_frames = 0
                self.enroll_used = 0
            else:
                self.enroll_name_edit = self.handle_text_input(key, self.enroll_name_edit, max_len=24)

        # cancel button
        cancel = Button("Cancel", (28, CANVAS_H - 72, 300, 52), tag="cancel")
        cancel.hover = point_in(self.mouse_x, self.mouse_y, cancel.rect)
        draw_button(canvas, cancel, primary=False)

        clicked = self.consume_click()
        if clicked:
            if point_in(self.mouse_x, self.mouse_y, name_rect) and self.enroll_name == "":
                self.focus_field = "live_enroll_name"
            elif cancel.hover:
                self.screen = "main"
                return

        if self.enroll_name == "":
            return

        # steps
        step_text, step_dur = ENROLL_STEPS[self.enroll_step_idx]
        elapsed_step = time.time() - self.enroll_step_t0

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

        if elapsed_step >= step_dur:
            self.enroll_step_idx += 1
            self.enroll_step_t0 = time.time()

            if self.enroll_step_idx >= len(ENROLL_STEPS):
                embs = np.array(self.enroll_embs, dtype=np.float32)
                protos, thr = build_profile_from_enrollment(self.enroll_name, embs)
                prof = MarkerProfile(
                    marker_id=self.enroll_name,
                    proto=protos,
                    thr=thr,
                    enroll_frames=int(self.enroll_frames),
                    enroll_used=int(self.enroll_used),
                    source_mode="webcam",
                    source_path="",
                )
                if self.registry is None:
                    self.registry = Registry()
                self.registry.add_marker(prof)
                self.save_registry()
                self.screen = "main"
                return

        # step card
        self.draw_sidebar_card(
            canvas,
            "Enrollment",
            [
                f"Name: {self.enroll_name}",
                f"Step: {step_text}",
                f"Embeddings used: {self.enroll_used}",
            ],
            y=235,
            h=150,
        )

        # progress bar
        total = sum(d for _, d in ENROLL_STEPS)
        elapsed_total = time.time() - self.enroll_t0
        prog = min(1.0, elapsed_total / max(total, 1e-6))
        x1, y1, x2, y2 = 28, 395, 28 + 300, 417
        rounded_rect(canvas, x1, y1, x2, y2, r=12, color=CARD2, thickness=-1)
        fill = int((x2 - x1) * prog)
        if fill > 0:
            rounded_rect(canvas, x1, y1, x1 + fill, y2, r=12, color=ACCENT, thickness=-1)
        draw_text(canvas, f"{int(elapsed_total)}/{int(total)}s", 28 + 210, 389, scale=0.50, color=TEXT, thick=2)

        # draw box
        if best_box is not None:
            x1i, y1i, x2i, y2i = best_box
            cx1 = int(offx + x1i * s)
            cy1 = int(offy + y1i * s)
            cx2 = int(offx + x2i * s)
            cy2 = int(offy + y2i * s)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), OK, 2)

    # ---------------- LIVE TRIALS (webcam) ----------------
    def start_live_trial_setup(self):
        self.screen = "live_trial_setup"
        self.trial_type = "swipe"
        self.condition_near = True
        self.condition_bright = True
        self.gt_enabled = True
        # force webcam
        self.source.open_webcam(self.source.webcam_index)

    def live_trial_setup_screen(self, canvas):
        self.draw_topbar(canvas, "Live Trials", "Choose type and condition, then start the live run.")

        if self.registry is None:
            self.draw_sidebar_card(canvas, "Error", ["No registry loaded.", "Run enrollment first."], y=92, h=90)
            back = Button("Back", (28, CANVAS_H - 72, 300, 52), tag="back")
            back.hover = point_in(self.mouse_x, self.mouse_y, back.rect)
            draw_button(canvas, back, primary=False)
            if self.consume_click() and back.hover:
                self.screen = "main"
            return

        shadow_card(canvas, 18, 92, 18 + 324, 92 + 290, r=16, shadow=8)
        draw_text(canvas, "Trial type", 34, 126, scale=0.58, color=ACCENT_HI, thick=2, font=FONT_TITLE)

        t_swipe = Button("Swipe", (34, 148, 300, 48), tag="t_swipe")
        t_int = Button("Interference", (34, 206, 300, 48), tag="t_int")
        t_re = Button("Re-entry", (34, 264, 300, 48), tag="t_re")

        t_swipe.toggled = (self.trial_type == "swipe")
        t_int.toggled = (self.trial_type == "interfere")
        t_re.toggled = (self.trial_type == "reentry")

        for b in (t_swipe, t_int, t_re):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        draw_text(canvas, "Condition", 34, 332, scale=0.58, color=ACCENT_HI, thick=2, font=FONT_TITLE)
        near_btn = Button("Near", (34, 354, 140, 48), tag="near")
        far_btn = Button("Far", (194, 354, 140, 48), tag="far")
        bright_btn = Button("Bright", (34, 412, 140, 48), tag="bright")
        dim_btn = Button("Dim", (194, 412, 140, 48), tag="dim")

        near_btn.toggled = self.condition_near
        far_btn.toggled = not self.condition_near
        bright_btn.toggled = self.condition_bright
        dim_btn.toggled = not self.condition_bright

        for b in (near_btn, far_btn, bright_btn, dim_btn):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        gt_btn = Button("GT ON" if self.gt_enabled else "GT OFF", (34, 474, 300, 48), tag="gt")
        gt_btn.toggled = self.gt_enabled
        gt_btn.hover = point_in(self.mouse_x, self.mouse_y, gt_btn.rect)
        draw_button(canvas, gt_btn, primary=True)

        back = Button("Back", (28, CANVAS_H - 72, 140, 52), tag="back")
        start = Button("Start", (188, CANVAS_H - 72, 140, 52), tag="start")
        for b in (back, start):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=(b.tag == "start"))

        clicked = self.consume_click()
        if clicked:
            if t_swipe.hover:
                self.trial_type = "swipe"
            elif t_int.hover:
                self.trial_type = "interfere"
            elif t_re.hover:
                self.trial_type = "reentry"
            elif near_btn.hover:
                self.condition_near = True
            elif far_btn.hover:
                self.condition_near = False
            elif bright_btn.hover:
                self.condition_bright = True
            elif dim_btn.hover:
                self.condition_bright = False
            elif gt_btn.hover:
                self.gt_enabled = not self.gt_enabled
            elif back.hover:
                self.screen = "main"
            elif start.hover:
                self.start_trial_run(mode="live")

    # ---------------- DEMO MENU ----------------
    def demo_menu_screen(self, canvas):
        self.draw_topbar(canvas, "Demo Mode", "Use video files for enrollment and trials. Paste a path or browse.")

        self.draw_sidebar_card(canvas, "Demo", ["Enrollment: build prototypes from a video", "Trials: run routing on a video"], y=92, h=90)

        btns = []
        bx, by, bw, bh, gap = 28, 205, 300, 54, 12

        def add(text, tag, enabled=True):
            b = Button(text=text, rect=(bx, by + len(btns) * (bh + gap), bw, bh), enabled=enabled, tag=tag)
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            btns.append(b)

        add("Demo Enrollment (Video)", "demo_enroll", enabled=True)
        add("Demo Trials (Video)", "demo_trials", enabled=True)
        add("Back", "back", enabled=True)

        for b in btns:
            draw_button(canvas, b, primary=(b.tag != "back"))

        clicked = self.consume_click()
        if clicked:
            for b in btns:
                if b.enabled and b.hover:
                    if b.tag == "demo_enroll":
                        self.start_demo_enroll_setup()
                    elif b.tag == "demo_trials":
                        self.start_demo_trial_setup()
                    elif b.tag == "back":
                        self.screen = "main"
                    break

    # ---------------- DEMO ENROLLMENT ----------------
    def start_demo_enroll_setup(self):
        self.screen = "demo_enroll_setup"
        self.focus_field = "demo_enroll_name"
        self.demo_enroll_name = ""
        self.demo_enroll_name_edit = ""
        self.demo_enroll_video = ""
        self.demo_enroll_video_edit = ""

    def demo_enroll_setup_screen(self, canvas, key):
        self.draw_topbar(canvas, "Demo Enrollment", "1) Name the marker. 2) Pick or paste a video. 3) Start.")

        shadow_card(canvas, 18, 92, 18 + 324, 92 + 290, r=16, shadow=8)
        name_rect = (34, 140, 300, 46)
        video_rect = (34, 216, 300, 46)

        name_value = self.demo_enroll_name_edit if self.demo_enroll_name == "" else self.demo_enroll_name
        draw_input(
            canvas,
            name_rect,
            "Marker name",
            name_value,
            placeholder="Type a name and press Enter",
            active=(self.focus_field == "demo_enroll_name"),
        )

        vraw = self.demo_enroll_video_edit if self.demo_enroll_video_edit else self.demo_enroll_video
        vshown = truncate_path(vraw)
        draw_input(
            canvas,
            video_rect,
            "Enrollment video",
            vshown,
            placeholder="Browse or paste a file path",
            active=(self.focus_field == "demo_enroll_video"),
        )

        pick = Button("Browse", (34, 274, 180, 46), tag="pick", enabled=True)
        paste = Button("Paste", (34 + 190, 274, 110, 46), tag="paste", enabled=True)
        for b in (pick, paste):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        start = Button("Start demo enrollment", (34, 330, 300, 48), tag="start",
                       enabled=bool(self.demo_enroll_name) and os.path.isfile(self.demo_enroll_video))
        start.hover = point_in(self.mouse_x, self.mouse_y, start.rect)
        draw_button(canvas, start, primary=True)

        back = Button("Back", (34, CANVAS_H - 72, 300, 52), tag="back")
        back.hover = point_in(self.mouse_x, self.mouse_y, back.rect)
        draw_button(canvas, back, primary=False)

        if key != -1:
            if self.focus_field == "demo_enroll_name":
                if key in (13, 10):
                    nm = self.demo_enroll_name_edit.strip()
                    self.demo_enroll_name = nm if nm else f"marker_{now_ms()}"
                    self.demo_enroll_name_edit = ""
                    self.focus_field = "demo_enroll_video"
                else:
                    self.demo_enroll_name_edit = self.handle_text_input(key, self.demo_enroll_name_edit, max_len=24)
            elif self.focus_field == "demo_enroll_video":
                if key not in (13, 10):
                    self.demo_enroll_video_edit = self.handle_text_input(key, self.demo_enroll_video_edit, max_len=220)
                    candidate = normalize_path(self.demo_enroll_video_edit)
                    if os.path.isfile(candidate):
                        candidate = os.path.abspath(candidate)
                        self.demo_enroll_video = candidate
                        self.demo_enroll_video_edit = candidate

        clicked = self.consume_click()
        if clicked:
            if point_in(self.mouse_x, self.mouse_y, name_rect):
                if self.demo_enroll_name:
                    self.demo_enroll_name_edit = self.demo_enroll_name
                    self.demo_enroll_name = ""
                self.focus_field = "demo_enroll_name"
            elif point_in(self.mouse_x, self.mouse_y, video_rect):
                self.focus_field = "demo_enroll_video"
            elif pick.hover:
                chosen = pick_video_file_windows(title="Select enrollment video")
                chosen = normalize_path(chosen)
                if chosen and os.path.isfile(chosen):
                    chosen = os.path.abspath(chosen)
                    self.demo_enroll_video = chosen
                    self.demo_enroll_video_edit = chosen
            elif paste.hover:
                clip = normalize_path(sanitize_clipboard_text(get_clipboard_text()))
                if clip:
                    self.demo_enroll_video_edit = clip
                    if os.path.isfile(clip):
                        clip = os.path.abspath(clip)
                        self.demo_enroll_video = clip
                        self.demo_enroll_video_edit = clip
            elif start.enabled and start.hover:
                self.start_demo_enroll_run()
            elif back.hover:
                self.screen = "demo_menu"

    def start_demo_enroll_run(self):
        # open video for demo enrollment
        self.source.open_video(self.demo_enroll_video, loop=True)
        self.screen = "demo_enroll_run"
        self.enroll_name = self.demo_enroll_name
        self.enroll_step_idx = 0
        self.enroll_step_t0 = time.time()
        self.enroll_t0 = time.time()
        self.enroll_embs = []
        self.enroll_frames = 0
        self.enroll_used = 0

    def demo_enroll_run_screen(self, frame_bgr, canvas, s, offx, offy):
        self.draw_topbar(canvas, "Demo Enrollment Running", "Enrollment is running on your video. Cancel to stop.")

        # reuse the same enrollment capture as live, with source_mode/video info
        step_text, step_dur = ENROLL_STEPS[self.enroll_step_idx]
        elapsed_step = time.time() - self.enroll_step_t0

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

        if elapsed_step >= step_dur:
            self.enroll_step_idx += 1
            self.enroll_step_t0 = time.time()

            if self.enroll_step_idx >= len(ENROLL_STEPS):
                embs = np.array(self.enroll_embs, dtype=np.float32)
                protos, thr = build_profile_from_enrollment(self.enroll_name, embs)
                prof = MarkerProfile(
                    marker_id=self.enroll_name,
                    proto=protos,
                    thr=thr,
                    enroll_frames=int(self.enroll_frames),
                    enroll_used=int(self.enroll_used),
                    source_mode="video",
                    source_path=self.demo_enroll_video,
                )
                if self.registry is None:
                    self.registry = Registry()
                self.registry.add_marker(prof)
                self.save_registry()

                # go back to demo menu and reopen webcam so UI feels "live"
                self.source.open_webcam(self.source.webcam_index)
                self.screen = "demo_menu"
                return

        self.draw_sidebar_card(
            canvas,
            "Demo Enrollment",
            [
                f"Name: {self.enroll_name}",
                f"Step: {step_text}",
                f"Embeddings used: {self.enroll_used}",
            ],
            y=92,
            h=150,
        )

        # progress bar
        total = sum(d for _, d in ENROLL_STEPS)
        elapsed_total = time.time() - self.enroll_t0
        prog = min(1.0, elapsed_total / max(total, 1e-6))
        x1, y1, x2, y2 = 28, 260, 28 + 300, 282
        rounded_rect(canvas, x1, y1, x2, y2, r=12, color=CARD2, thickness=-1)
        fill = int((x2 - x1) * prog)
        if fill > 0:
            rounded_rect(canvas, x1, y1, x1 + fill, y2, r=12, color=ACCENT, thickness=-1)
        draw_text(canvas, f"{int(elapsed_total)}/{int(total)}s", 28 + 210, 254, scale=0.50, color=TEXT, thick=2)

        if best_box is not None:
            x1i, y1i, x2i, y2i = best_box
            cx1 = int(offx + x1i * s)
            cy1 = int(offy + y1i * s)
            cx2 = int(offx + x2i * s)
            cy2 = int(offy + y2i * s)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), OK, 2)

        cancel = Button("Cancel", (28, CANVAS_H - 72, 300, 52), tag="cancel")
        cancel.hover = point_in(self.mouse_x, self.mouse_y, cancel.rect)
        draw_button(canvas, cancel, primary=False)

        if self.consume_click() and cancel.hover:
            self.source.open_webcam(self.source.webcam_index)
            self.screen = "demo_menu"

    # ---------------- DEMO TRIALS ----------------
    def start_demo_trial_setup(self):
        self.screen = "demo_trial_setup"
        self.focus_field = "demo_trial_video"
        self.demo_trial_video = ""
        self.demo_trial_video_edit = ""
        self.demo_trial_type = "swipe"
        self.demo_condition_near = True
        self.demo_condition_bright = True
        self.demo_gt_enabled = False

    def demo_trial_setup_screen(self, canvas, key):
        self.draw_topbar(canvas, "Demo Trials", "Pick a registry (optional), choose a video, then start the trial.")

        if self.registry is None:
            self.draw_sidebar_card(canvas, "Registry", ["No registry loaded.", "Load registry.json or run enrollment."], y=92, h=90)
        else:
            self.draw_sidebar_card(canvas, "Registry", ["Markers: " + ", ".join(self.registry.names())], y=92, h=90)

        # left controls
        shadow_card(canvas, 18, 205, 18 + 324, 205 + 360, r=16, shadow=8)
        draw_text(canvas, "Trial type", 34, 238, scale=0.58, color=ACCENT_HI, thick=2, font=FONT_TITLE)

        t_swipe = Button("Swipe", (34, 262, 300, 46), tag="t_swipe")
        t_int = Button("Interference", (34, 318, 300, 46), tag="t_int")
        t_re = Button("Re-entry", (34, 374, 300, 46), tag="t_re")

        t_swipe.toggled = (self.demo_trial_type == "swipe")
        t_int.toggled = (self.demo_trial_type == "interfere")
        t_re.toggled = (self.demo_trial_type == "reentry")

        for b in (t_swipe, t_int, t_re):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        draw_text(canvas, "Condition", 34, 438, scale=0.58, color=ACCENT_HI, thick=2, font=FONT_TITLE)
        near_btn = Button("Near", (34, 460, 140, 46), tag="near")
        far_btn = Button("Far", (194, 460, 140, 46), tag="far")
        bright_btn = Button("Bright", (34, 516, 140, 46), tag="bright")
        dim_btn = Button("Dim", (194, 516, 140, 46), tag="dim")

        near_btn.toggled = self.demo_condition_near
        far_btn.toggled = not self.demo_condition_near
        bright_btn.toggled = self.demo_condition_bright
        dim_btn.toggled = not self.demo_condition_bright

        for b in (near_btn, far_btn, bright_btn, dim_btn):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        # right: file selection
        shadow_card(canvas, 360, 92, CANVAS_W - 18, 92 + 250, r=18, shadow=10)
        draw_text(canvas, "Trial video", 382, 126, scale=0.62, color=ACCENT_HI, thick=2, font=FONT_TITLE)
        draw_text(canvas, "Pick a file or paste a path, then start the trial.", 382, 150, scale=0.5, color=MUTED, thick=2)

        vraw = self.demo_trial_video_edit if self.demo_trial_video_edit else self.demo_trial_video
        vshown = truncate_path(vraw)
        video_rect = (382, 170, CANVAS_W - 18 - 382 - 20, 46)
        draw_input(
            canvas,
            video_rect,
            "Video path",
            vshown,
            placeholder="Browse or paste a file path",
            active=(self.focus_field == "demo_trial_video"),
        )

        btn_x = 382
        btn_y = 230
        btn_w = 185
        gap = 10
        pickv = Button("Browse", (btn_x, btn_y, btn_w, 46), tag="pickv")
        paste = Button("Paste", (btn_x + (btn_w + gap) * 1, btn_y, btn_w, 46), tag="paste")
        loadreg = Button("Pick registry", (btn_x + (btn_w + gap) * 2, btn_y, btn_w, 46), tag="pickr")
        reload_btn = Button("Reload registry", (btn_x + (btn_w + gap) * 3, btn_y, btn_w, 46), tag="reload")

        for b in (pickv, paste, loadreg, reload_btn):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=True)

        if key != -1 and self.focus_field == "demo_trial_video" and key not in (13, 10):
            self.demo_trial_video_edit = self.handle_text_input(key, self.demo_trial_video_edit, max_len=240)
            candidate = normalize_path(self.demo_trial_video_edit)
            if os.path.isfile(candidate):
                candidate = os.path.abspath(candidate)
                self.demo_trial_video = candidate
                self.demo_trial_video_edit = candidate

        # bottom
        back = Button("Back", (28, CANVAS_H - 72, 180, 52), tag="back")
        start = Button("Start demo trial", (CANVAS_W - 18 - 260, CANVAS_H - 72, 260, 52), tag="start",
                       enabled=(self.registry is not None) and os.path.isfile(self.demo_trial_video))
        for b in (back, start):
            b.hover = point_in(self.mouse_x, self.mouse_y, b.rect)
            draw_button(canvas, b, primary=(b.tag == "start"))

        clicked = self.consume_click()
        if clicked:
            if t_swipe.hover:
                self.demo_trial_type = "swipe"
            elif t_int.hover:
                self.demo_trial_type = "interfere"
            elif t_re.hover:
                self.demo_trial_type = "reentry"
            elif near_btn.hover:
                self.demo_condition_near = True
            elif far_btn.hover:
                self.demo_condition_near = False
            elif bright_btn.hover:
                self.demo_condition_bright = True
            elif dim_btn.hover:
                self.demo_condition_bright = False
            elif point_in(self.mouse_x, self.mouse_y, video_rect):
                self.focus_field = "demo_trial_video"
            elif pickv.hover:
                chosen = pick_video_file_windows(title="Select trial video")
                chosen = normalize_path(chosen)
                if chosen and os.path.isfile(chosen):
                    chosen = os.path.abspath(chosen)
                    self.demo_trial_video = chosen
                    self.demo_trial_video_edit = chosen
            elif paste.hover:
                clip = normalize_path(sanitize_clipboard_text(get_clipboard_text()))
                if clip:
                    self.demo_trial_video_edit = clip
                    if os.path.isfile(clip):
                        clip = os.path.abspath(clip)
                        self.demo_trial_video = clip
                        self.demo_trial_video_edit = clip
            elif loadreg.hover:
                chosen = pick_registry_file_windows(title="Select registry.json")
                chosen = normalize_path(chosen)
                if chosen and os.path.isfile(chosen):
                    try:
                        with open(chosen, "r", encoding="utf-8") as f:
                            r = Registry.from_json(json.load(f))
                        if len(r.markers) > 0:
                            self.registry = r
                            # also write into default reg_path for consistency
                            self.save_registry()
                    except Exception:
                        pass
            elif reload_btn.hover:
                self.load_registry()
            elif back.hover:
                self.screen = "demo_menu"
            elif start.enabled and start.hover:
                self.start_trial_run(mode="demo")

    # ---------------- TRIAL RUN (shared) ----------------
    def start_trial_run(self, mode: str):
        assert self.registry is not None

        # set parameters based on mode
        if mode == "live":
            self.trial_type = self.trial_type
            self.condition_near = self.condition_near
            self.condition_bright = self.condition_bright
            self.gt_enabled = self.gt_enabled
            self.source.open_webcam(self.source.webcam_index)
        else:
            self.trial_type = self.demo_trial_type
            self.condition_near = self.demo_condition_near
            self.condition_bright = self.demo_condition_bright
            self.gt_enabled = False  # keep demo simple unless you want GT UI too
            self.source.open_video(self.demo_trial_video, loop=True)

        if self.trial_type == "swipe":
            self.trial_duration = TRIAL_SWIPE_SEC
        elif self.trial_type == "interfere":
            self.trial_duration = TRIAL_INTERFERE_SEC
        else:
            self.trial_duration = TRIAL_REENTRY_SEC

        self.screen = "live_trial_run" if mode == "live" else "demo_trial_run"
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

        self._cm = np.zeros((len(self.registry.names()), len(self.registry.names())), dtype=np.int32)
        self._gt_total = 0
        self._gt_correct = 0

        # reset Ultralytics trackers between runs if possible
        try:
            if hasattr(self.yolo, "predictor") and self.yolo.predictor is not None:
                if hasattr(self.yolo.predictor, "trackers"):
                    del self.yolo.predictor.trackers
                if hasattr(self.yolo.predictor, "vid_path"):
                    del self.yolo.predictor.vid_path
        except Exception:
            pass

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
            "gt_samples": int(self._gt_total),
            "top1_accuracy": (float(self._gt_correct / self._gt_total) if self._gt_total > 0 else None),
        }

        if self._gt_total > 0 and self.registry is not None:
            cm_path = os.path.join(out_trial_dir, "confusion_matrix.png")
            save_confusion_matrix(self.registry.names(), self._cm, cm_path)
            summary["confusion_matrix_png"] = cm_path
            summary["confusion_matrix"] = self._cm.tolist()
            summary["classes"] = self.registry.names()

        summ_path = os.path.join(out_trial_dir, "trial_summary.json")
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # back to appropriate menu
        self.source.open_webcam(self.source.webcam_index)
        if self.screen == "demo_trial_run":
            self.screen = "demo_menu"
        else:
            self.screen = "main"

    def trial_run_tick(self, frame_bgr, canvas, s, offx, offy, view_rect):
        assert self.registry is not None

        t_frame0 = time.time()
        self.frame_idx += 1

        t_det0 = time.time()
        try:
            if hasattr(self.yolo, "predictor") and self.yolo.predictor is not None:
                if getattr(self.yolo.predictor, "trackers", "missing") is None:
                    del self.yolo.predictor.trackers
                    if hasattr(self.yolo.predictor, "vid_path"):
                        del self.yolo.predictor.vid_path
        except Exception:
            pass
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
        det_meta = []
        crops_rgb = []

        H, W = frame_bgr.shape[:2]
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(boxes),), dtype=np.float32)
            tids = boxes.id.cpu().numpy().astype(int) if (hasattr(boxes, "id") and boxes.id is not None) else -np.ones((len(boxes),), dtype=np.int32)

            order = np.argsort(-confs)
            for i in order:
                tid = int(tids[i])
                x1, y1, x2, y2 = xyxy[i].tolist()
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

                det_meta.append((tid, (x1i, y1i, x2i, y2i)))
                crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        t_emb0 = time.time()
        Z = embed_crops(self.embedder, crops_rgb, DEVICE)
        embed_ms = (time.time() - t_emb0) * 1000.0

        t_match0 = time.time()
        preds = []
        active_now = set()

        for i in range(Z.shape[0]):
            tid, box = det_meta[i]
            z = Z[i]

            if tid < 0:
                pred, sim = match_marker(z, self.registry)
                preds.append((tid, pred, float(sim), box))
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
        for tid_lost in (self.active_prev - active_now):
            self.event_rows.append([self.trial_id, self.trial_type, self.condition_str(), now_ms(), "track_lost", tid_lost, "", "", ""])
        self.active_prev = active_now

        unknown_tracks = sum(1 for (_, pred, _, _) in preds if pred == "unknown")

        remaining = max(0.0, self.trial_duration - (time.time() - self.trial_t0))
        self.draw_sidebar_card(
            canvas,
            "Trial HUD",
            [
                f"Trial: {self.trial_type}",
                f"Condition: {self.condition_str()}",
                f"Remaining: {remaining:.1f}s",
                f"Dets: {len(preds)} Tracks: {len(active_now)}",
                f"Latency(ms): det {det_track_ms:.1f} emb {embed_ms:.1f} match {match_ms:.1f}",
            ],
            y=92,
            h=150,
        )

        # boxes
        clickable_boxes = []
        for (tid, pred, sim, box) in preds:
            x1i, y1i, x2i, y2i = box
            cx1 = int(offx + x1i * s)
            cy1 = int(offy + y1i * s)
            cx2 = int(offx + x2i * s)
            cy2 = int(offy + y2i * s)

            col = OK if pred != "unknown" else (120, 170, 255)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), col, 2)
            draw_text(canvas, f"T{tid} {pred} {sim:.2f}", cx1, max(20, cy1 - 8), scale=0.50, color=WHITE, thick=2)
            clickable_boxes.append((tid, (cx1, cy1, cx2 - cx1, cy2 - cy1)))

        stop = Button("End trial", (28, CANVAS_H - 72, 300, 52), tag="stop")
        stop.hover = point_in(self.mouse_x, self.mouse_y, stop.rect)
        draw_button(canvas, stop, primary=True)

        clicked = self.consume_click()
        if clicked and stop.hover:
            self.finish_trial()
            return

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

        if remaining <= 0.0:
            self.finish_trial()

    # ---------------- MAIN TICK ----------------
    def tick(self, frame_bgr, key):
        canvas, s, offx, offy, view_rect = compose_canvas(frame_bgr)
        draw_text(canvas, "ESC/Q: exit", 380, CANVAS_H - 14, scale=0.46, color=MUTED, thick=2)

        if self.screen == "main":
            self.main_screen(canvas)
        elif self.screen == "live_enroll":
            self.live_enroll_screen(frame_bgr, canvas, s, offx, offy, key)
        elif self.screen == "live_trial_setup":
            self.live_trial_setup_screen(canvas)
        elif self.screen == "live_trial_run":
            self.draw_topbar(canvas, "Live Trial Running", "Routing is running from webcam. End anytime.")
            self.trial_run_tick(frame_bgr, canvas, s, offx, offy, view_rect)
        elif self.screen == "demo_menu":
            self.demo_menu_screen(canvas)
        elif self.screen == "demo_enroll_setup":
            self.demo_enroll_setup_screen(canvas, key)
        elif self.screen == "demo_enroll_run":
            self.demo_enroll_run_screen(frame_bgr, canvas, s, offx, offy)
        elif self.screen == "demo_trial_setup":
            self.demo_trial_setup_screen(canvas, key)
        elif self.screen == "demo_trial_run":
            self.draw_topbar(canvas, "Demo Trial Running", "Routing is running.")
            self.trial_run_tick(frame_bgr, canvas, s, offx, offy, view_rect)

        cv2.imshow(self.win, canvas)

    def run(self):
        while self.running:
            if window_closed(self.win):
                break

            ok, frame = self.source.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

            # route to correct setup/run screens
            if self.screen == "live_trials":
                self.screen = "live_trial_setup"

            self.tick(frame, key)

        self.running = False


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(0)
    ensure_dir(RUN_DIR)

    if not os.path.isfile(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO_WEIGHTS not found: {YOLO_WEIGHTS}")
    if not os.path.isfile(EMBEDDER_WEIGHTS):
        raise FileNotFoundError(f"EMBEDDER_WEIGHTS not found: {EMBEDDER_WEIGHTS}")

    yolo = YOLO(YOLO_WEIGHTS)
    embedder = load_embedder(EMBEDDER_WEIGHTS, DEVICE)

    app = PINCHApp(yolo, embedder)
    app.run()

    try:
        app.source.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()