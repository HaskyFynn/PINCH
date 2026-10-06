"""Model loading and batched features; no camera or UI side effects on import."""
import hashlib
from pathlib import Path
import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / max(min(area_a, area_b), 1e-6)


def box_iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0.,x[2]-x[0])*max(0.,x[3]-x[1])
    return inter/max(area(a)+area(b)-inter,1e-9)


def crop_quality(frame, box, settings):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    px, py = (x2-x1)*settings.crop_padding, (y2-y1)*settings.crop_padding
    a, b, c, d = max(0, int(x1-px)), max(0, int(y1-py)), min(w, int(x2+px)), min(h, int(y2+py))
    if c <= a or d <= b:
        return None, 'invalid_crop', 0.
    crop = frame[b:d, a:c]
    if (c-a)*(d-b) < settings.min_crop_area:
        return None, 'too_small', 0.
    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if blur < settings.blur_threshold:
        return None, 'blurred', blur
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 'usable', blur


class Models:
    def __init__(self, settings):
        import os
        Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import models, transforms
        from ultralytics import YOLO
        from PIL import Image

        for name in ('detector', 'embedder'):
            if not settings.path(name).is_file():
                raise FileNotFoundError(f'Missing {name}: {settings.path(name)}')
        torch.set_num_threads(settings.cpu_threads)
        self.device = settings.device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this Python environment; select cpu in config/live.json')

        class ResnetEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                # The complete checkpoint is supplied. Never download ImageNet
                # weights just to overwrite them with the trained PINCH model.
                m = models.resnet18(weights=None)
                n = m.fc.in_features
                m.fc = nn.Identity()
                self.backbone = m
                self.proj = nn.Linear(n, 128)

            def forward(self, x):
                return F.normalize(self.proj(self.backbone(x)), p=2, dim=1)

        self.yolo = YOLO(str(settings.path('detector')))
        self.embedder = ResnetEmbedder().to(self.device)
        checkpoint = torch.load(settings.path('embedder'), map_location=self.device, weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        self.embedder.load_state_dict(state, strict=True)
        self.embedder.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
        self.torch, self.Image = torch, Image
        self.settings = settings
        self.embedder_hash = sha256(settings.path('embedder'))
        self.detector_hash = sha256(settings.path('detector'))

    def detect(self, frame):
        s = self.settings
        result = self.yolo.predict(frame, conf=s.detector_confidence, iou=s.detector_iou,
                                   imgsz=s.image_size, max_det=s.max_detections,
                                   device=self.device, verbose=False)[0].boxes.cpu().numpy()
        # Ultralytics overrides PyTorch's thread count when its lazy predictor is
        # first created. Restore the configured budget before embedding crops.
        if self.device == 'cpu' and self.torch.get_num_threads() != s.cpu_threads:
            self.torch.set_num_threads(s.cpu_threads)
        return result

    def embed(self, crops):
        if not crops:
            return np.empty((0, 128), dtype=np.float32)
        torch = self.torch
        x = torch.stack([self.transform(self.Image.fromarray(c)) for c in crops]).to(self.device)
        with torch.inference_mode():
            z = self.embedder(x)
        return z.cpu().numpy().astype(np.float32)
