import time

from yamspy import MSPy

with MSPy(device="/dev/ttyACM0", baudrate=115200) as msp:
    # 4 motors: stop all
    motors = [1000] * 8
    msp.send_RAW_MOTORS(motors)

    time.sleep(0.2)

    # spin motor 1
    motors[0] = 1010
    msp.send_RAW_MOTORS(motors)

    time.sleep(1.5)

    # stop again
    motors = [1000] * 8
    msp.send_RAW_MOTORS(motors)
