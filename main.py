import threading
from queue import Queue

import cv2

from config import AppConfig
from motion_detector import QuadMotionDetector
from motion_result import MotionResult
from motor_runner import MotorRunner


def make_motion_queue(maxsize: int = 1) -> Queue[MotionResult]:
    # maxsize=1 = "latest value only" queue
    return Queue(maxsize=maxsize)


def main():
    cfg = AppConfig(
        window_name="motion_debug",
        width=1280,
        height=720,
        fps=30,
        device_index=0,
        v4l2_device=None,  # set "/dev/video0" to force
        mirror_h=True,
        mirror_v=False,
        window_seconds=0.15
    )

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_name, cfg.width, cfg.height)
    print("Controls: [q]=quit  [space]=pause/resume  [r]=reset background(prev frame)")

    queue = make_motion_queue(maxsize=1)
    stop_event = threading.Event()

    motion_detector = QuadMotionDetector(cfg, queue, stop_event)
    motor_runner = MotorRunner(queue, stop_event)

    t_detector = threading.Thread(target=motion_detector.run_detection(), name="quad-motion-detector")
    t_runner = threading.Thread(target=motor_runner.run_motor_check(), name="motor-runner")

    t_detector.start()
    t_runner.start()

    # while True:
    #     # motion_vis = (accum_motion > 0).astype(np.uint8) * 255
    #     # vis = overlay_motion(frame, motion_vis, cfg.overlay_alpha)
    #     draw_rois(frame, rois, winner)
    #
    #     cv2.putText(
    #         frame,
    #         f"moving={winner or 'NONE'}  isolation={isolation:.2f}  total={int(total)}",
    #         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    #     )
    #     # cv2.putText(
    #     #     frame,
    #     #     "  ".join([f"{k}:{int(v)}" for k, v in accum_scores.items()]),
    #     #     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2
    #     # )
    #
    #     cv2.imshow(cfg.window_name, frame)

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break
    # if key == ord(' '):
    #     paused = not paused
    # if key == ord('r'):
    #     self.reset()

    try:
        t_detector.join()
        t_runner.join()
    finally:
        stop_event.set()
        t_detector.join()
        t_runner.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
