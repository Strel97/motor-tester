import threading
from queue import Queue

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

    queue = make_motion_queue(maxsize=1)
    stop_event = threading.Event()

    motion_detector = QuadMotionDetector(cfg, queue, stop_event)
    motor_runner = MotorRunner(queue, stop_event)

    t_detector = threading.Thread(target=motion_detector.run_detection, name="quad-motion-detector")
    t_runner = threading.Thread(target=motor_runner.run_motor_check, name="motor-runner")

    t_detector.start()
    t_runner.start()

    try:
        t_detector.join()
        t_runner.join()
    finally:
        stop_event.set()
        t_detector.join()
        t_runner.join()


if __name__ == "__main__":
    main()
