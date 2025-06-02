
#!/usr/bin/env python3

import time
from ev3dev2.motor import SpeedDPS, LargeMotor, OUTPUT_B, OUTPUT_C


motor_right = LargeMotor(OUTPUT_C)
motor_left = LargeMotor(OUTPUT_B)

motor_right.stop()
motor_left.stop()