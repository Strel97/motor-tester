import time
from queue import Queue
from threading import Event

from yamspy import MSPy

from motion_result import MotionResult


class MotorRunner:
    correct_order = [MotionResult.RR, MotionResult.FR, MotionResult.RL, MotionResult.FL]

    def __init__(self, motion_result: Queue[MotionResult], stop_event: Event):
        self.motion_result = motion_result
        self.stop_event = stop_event

    @staticmethod
    def _stop_all_motors(msp: MSPy):
        motors = [1000] * 8
        msp.send_RAW_MOTORS(motors)

    @staticmethod
    def _run_motor(msp: MSPy, motor_index: int):
        # spin motor 1
        motors = [1000] * 8
        motors[motor_index] = 1010
        msp.send_RAW_MOTORS(motors)

    def run_motor_check(self):
        with MSPy(device="/dev/ttyACM0", baudrate=115200) as msp:
            self._stop_all_motors(msp)

            time.sleep(2)

            for motor_index in range(4):
                self._run_motor(msp, motor_index)
                res: MotionResult = self.motion_result.get()
                if res != self.correct_order[motor_index]:
                    print(f"Motor {motor_index + 1} is {res.name} wrong!!!")
                time.sleep(1.5)

            self._stop_all_motors(msp)
            self.stop_event.set()
