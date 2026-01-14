from typing import Protocol, Tuple, Optional

import cv2
import numpy as np

from config import AppConfig


class FrameSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...

    def stop(self) -> None: ...

    def name(self) -> str: ...


class OpenCVCamera(FrameSource):
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None

    def name(self) -> str:
        return "OpenCV VideoCapture"

    def start(self) -> None:
        # Prefer path if provided, otherwise index.
        if self.cfg.v4l2_device:
            self.cap = cv2.VideoCapture(self.cfg.v4l2_device, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(self.cfg.device_index)

        if not self.cap.isOpened():
            raise RuntimeError("OpenCV camera could not be opened")

        # Apply conservative settings; avoid FPS on problematic drivers
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        # On some devices FPS setting breaks; enable only if needed:
        # self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PiCamera2Source(FrameSource):
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.picam2 = None

    def name(self) -> str:
        return "Picamera2"

    def start(self) -> None:
        from picamera2 import Picamera2  # import here so PC doesn't need it
        self.picam2 = Picamera2()

        # Prefer BGR888; fallback to XRGB8888 (BGRA) if needed
        try:
            config = self.picam2.create_video_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "BGR888"},
                controls={"FrameRate": self.cfg.fps},
            )
        except Exception:
            config = self.picam2.create_video_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "XRGB8888"},
                controls={"FrameRate": self.cfg.fps},
            )

        self.picam2.configure(config)
        self.picam2.start()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        assert self.picam2 is not None
        frame = self.picam2.capture_array()
        if frame is None or frame.size == 0:
            return False, None
        # Handle 4-channel from XRGB8888
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, frame

    def stop(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                try:
                    self.picam2.close()
                except Exception:
                    pass
            self.picam2 = None


def make_camera(cfg: AppConfig) -> FrameSource:
    import importlib.util
    if importlib.util.find_spec("picamera2") is not None:
        return PiCamera2Source(cfg)
    return OpenCVCamera(cfg)
