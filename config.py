from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    window_name: str
    width: int = 1280
    height: int = 720
    fps: int = 30

    # Camera selection:
    # - On PC: device_index=0 usually
    # - On Pi: ignored if Picamera2 is used
    device_index: int = 0
    v4l2_device: Optional[str] = None  # e.g. "/dev/video0" to force by path

    mirror_h: bool = False
    mirror_v: bool = False

    blur_ksize: int = 5
    diff_thresh: int = 18
    window_seconds: float = 0.15  # for live debug; for motor-step measurement use ~0.8-1.2
    min_motion_sum: float = 2e6
    min_isolation: float = 1.8
    overlay_alpha: float = 0.55
