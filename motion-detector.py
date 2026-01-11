#!/usr/bin/env python3
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class DetectorConfig:
    blur_ksize: int = 5          # Gaussian blur kernel (odd). 0/1 disables blur
    diff_thresh: int = 18        # Threshold on absdiff for motion mask
    min_motion_sum: float = 2e6  # Minimum total motion to consider "something moving" (tune)
    min_isolation: float = 1.8   # best_score / second_best_score must exceed this
    window_seconds: float = 0.8  # accumulate motion over this window
    overlay_alpha: float = 0.55  # motion overlay transparency

def split_into_rois(frame_w: int, frame_h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Fixed split into 4 rectangles:
      TL, TR, BR, BL
    Labels are physical names as seen in the image.
    Change labels if your camera is rotated.
    """
    half_w = frame_w // 2
    half_h = frame_h // 2
    return {
        "FL": (0,        0,        half_w,             half_h),
        "FR": (half_w,   0,        frame_w - half_w,   half_h),
        "RR": (half_w,   half_h,   frame_w - half_w,   frame_h - half_h),
        "RL": (0,        half_h,   half_w,             frame_h - half_h),
    }

class QuadMotionDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.prev_gray: Optional[np.ndarray] = None

    def motion_mask(self, gray: np.ndarray) -> np.ndarray:
        """Return binary motion mask (uint8 0/255)."""
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.uint8)

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        if self.cfg.blur_ksize and self.cfg.blur_ksize >= 3:
            diff = cv2.GaussianBlur(diff, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        _, motion = cv2.threshold(diff, self.cfg.diff_thresh, 255, cv2.THRESH_BINARY)
        return motion

    def score_rois(self, motion: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, float]:
        scores = {}
        for label, (x, y, w, h) in rois.items():
            scores[label] = float(np.sum(motion[y:y+h, x:x+w]))
        return scores

    @staticmethod
    def pick_best(scores: Dict[str, float]) -> Tuple[str, float, float]:
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_val = items[0]
        second_val = items[1][1] if len(items) > 1 else 0.0
        isolation = (best_val / second_val) if second_val > 0 else float("inf")
        total = sum(scores.values())
        return best_label, isolation, total

def overlay_motion(frame_bgr: np.ndarray, motion_mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Overlay red highlight where motion_mask is 255.
    """
    overlay = frame_bgr.copy()
    # red layer where motion
    overlay[motion_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)

def draw_rois(frame: np.ndarray, rois: Dict[str, Tuple[int,int,int,int]], winner: Optional[str]) -> None:
    for label, (x, y, w, h) in rois.items():
        # Thicker border for winner
        thickness = 4 if (winner is not None and label == winner) else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness)
        cv2.putText(frame, label, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

def format_scores(scores: Dict[str, float]) -> str:
    return "  ".join([f"{k}:{int(v)}" for k, v in scores.items()])

def main():
    cfg = DetectorConfig(
        blur_ksize=5,
        diff_thresh=18,
        min_motion_sum=2e6,
        min_isolation=1.8,
        window_seconds=0.15,   # for live debug, short window feels responsive
        overlay_alpha=0.55
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Try another index or a GStreamer pipeline.")

    # Fix resolution for stable ROI splitting
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Determine actual frame size
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed.")
    h, w = frame.shape[:2]
    rois = split_into_rois(w, h)

    detector = QuadMotionDetector(cfg)

    paused = False
    print("Debug window controls: [q]=quit  [space]=pause/resume  [r]=reset background(prev frame)")
    cv2.namedWindow("motion_debug", cv2.WINDOW_NORMAL)

    # For live display we accumulate scores over a short rolling window.
    # We'll do it by summing N frames worth of motion masks.
    accum_scores = {k: 0.0 for k in rois.keys()}
    accum_motion = np.zeros((h, w), dtype=np.uint32)
    accum_frames = 0
    window_start = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = detector.motion_mask(gray)

                # accumulate
                scores = detector.score_rois(motion, rois)
                for k, v in scores.items():
                    accum_scores[k] += v
                accum_motion += (motion.astype(np.uint32))
                accum_frames += 1

                # If window elapsed, evaluate winner and reset accumulators
                now = time.time()
                if (now - window_start) >= cfg.window_seconds:
                    best, isolation, total = detector.pick_best(accum_scores)
                    winner = None
                    if total >= cfg.min_motion_sum and isolation >= cfg.min_isolation:
                        winner = best

                    # Build overlay from accumulated motion (normalize to 0/255 for display)
                    # Anything >0 means motion occurred at least once in the window.
                    motion_vis = (accum_motion > 0).astype(np.uint8) * 255

                    vis = overlay_motion(frame, motion_vis, cfg.overlay_alpha)
                    draw_rois(vis, rois, winner)

                    # Top HUD
                    cv2.putText(vis, f"moving={winner or 'NONE'}  isolation={isolation:.2f}  total={int(total)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    # Scores line
                    cv2.putText(vis, format_scores(accum_scores),
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

                    cv2.imshow("motion_debug", vis)

                    # reset for next window
                    accum_scores = {k: 0.0 for k in rois.keys()}
                    accum_motion[:] = 0
                    accum_frames = 0
                    window_start = now

            # Handle keys even when paused
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                paused = not paused
            elif k == ord('r'):
                detector.prev_gray = None  # reset background reference

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
