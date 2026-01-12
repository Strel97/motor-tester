#!/usr/bin/env python3
import time
from queue import Queue, Full
from threading import Event
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from camera_source import make_camera
from config import AppConfig
from motion_result import MotionResult


# ------------------ Config ------------------

def split_into_rois(w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    half_w, half_h = w // 2, h // 2
    return {
        "FL": (0, 0, half_w, half_h),
        "FR": (half_w, 0, w - half_w, half_h),
        "RR": (half_w, half_h, w - half_w, h - half_h),
        "RL": (0, half_h, half_w, h - half_h),
    }


def apply_flips(frame: np.ndarray, cfg: AppConfig) -> np.ndarray:
    if cfg.mirror_h and cfg.mirror_v:
        return cv2.flip(frame, -1)
    if cfg.mirror_h:
        return cv2.flip(frame, 1)
    if cfg.mirror_v:
        return cv2.flip(frame, 0)
    return frame


def overlay_motion(frame_bgr: np.ndarray, motion_mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = frame_bgr.copy()
    overlay[motion_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)


def draw_rois(frame: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]], winner: Optional[str]) -> None:
    for label, (x, y, w, h) in rois.items():
        thickness = 4 if (winner is not None and label == winner) else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# ------------------ Motion Detector ------------------

class QuadMotionDetector:
    def __init__(self, cfg: AppConfig, motion_result: Queue[MotionResult], stop_event: Event) -> None:
        self.cfg = cfg
        self.stop_event = stop_event
        self.motion_result = motion_result
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None

    def motion_mask(self, gray: np.ndarray) -> np.ndarray:
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.uint8)

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        if self.cfg.blur_ksize and self.cfg.blur_ksize >= 3:
            diff = cv2.GaussianBlur(diff, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        _, motion = cv2.threshold(diff, self.cfg.diff_thresh, 255, cv2.THRESH_BINARY)
        return motion

    def update_motion_result(self, result: MotionResult) -> None:
        try:
            self.motion_result.get_nowait()  # drop old value if present
        except Exception:
            pass
        try:
            self.motion_result.put_nowait(result)
        except Full:
            pass

    def run_detection(self):
        cam = make_camera(self.cfg)
        print("Selected camera backend:", cam.name())

        cam.start()

        # Prime and get frame size
        ok, frame = cam.read()
        if not ok:
            cam.stop()
            raise RuntimeError("Camera read failed at startup")

        frame = apply_flips(frame, self.cfg)
        h, w = frame.shape[:2]
        rois = split_into_rois(w, h)

        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, self.cfg.width, self.cfg.height)
        print("Controls: [q]=quit  [space]=pause/resume  [r]=reset background(prev frame)")

        paused = False
        accum_scores = {k: 0.0 for k in rois.keys()}
        accum_motion = np.zeros((h, w), dtype=np.uint32)
        window_start = time.time()

        try:
            while not self.stop_event.is_set():
                if not paused:
                    ok, frame = cam.read()
                    if not ok or frame is None:
                        # If PC webcam occasionally drops frames, just skip
                        continue

                    frame = apply_flips(frame, self.cfg)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    motion = self.motion_mask(gray)

                    # accumulate per ROI
                    for label, (x, y, ww, hh) in rois.items():
                        accum_scores[label] += float(np.sum(motion[y:y + hh, x:x + ww]))
                    accum_motion += motion.astype(np.uint32)

                    now = time.time()
                    if (now - window_start) >= self.cfg.window_seconds:
                        best, isolation, total = self.pick_best(accum_scores)
                        winner = None
                        if total >= self.cfg.min_motion_sum and isolation >= self.cfg.min_isolation:
                            winner = best
                            self.update_motion_result(MotionResult[winner])

                        # motion_vis = (accum_motion > 0).astype(np.uint8) * 255
                        # vis = overlay_motion(frame, motion_vis, cfg.overlay_alpha)
                        draw_rois(frame, rois, winner)

                        cv2.putText(
                            frame,
                            f"moving={winner or 'NONE'}  isolation={isolation:.2f}  total={int(total)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                        )
                        # cv2.putText(
                        #     frame,
                        #     "  ".join([f"{k}:{int(v)}" for k, v in accum_scores.items()]),
                        #     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2
                        # )

                        cv2.imshow(self.cfg.window_name, frame)
                        # reset window
                        accum_scores = {k: 0.0 for k in rois.keys()}
                        accum_motion[:] = 0
                        window_start = now

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' '):
                    paused = not paused
                if key == ord('r'):
                    self.reset()

        finally:
            cam.stop()
            cv2.destroyAllWindows()

    @staticmethod
    def pick_best(scores: Dict[str, float]) -> Tuple[str, float, float]:
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_val = items[0]
        second_val = items[1][1] if len(items) > 1 else 0.0
        isolation = (best_val / second_val) if second_val > 0 else float("inf")
        total = sum(scores.values())
        return best_label, isolation, total


def main():
    pass


if __name__ == "__main__":
    main()
