"""PINCH multi-marker enrollment and recognition."""

__version__ = "0.2.0"

# Keep runtime settings with the project and avoid dependency auto-installation
# during camera sessions. Dependencies are installed explicitly by the user.
import os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
