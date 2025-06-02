
#!/usr/bin/env python3

import time
from ev3dev2.motor import SpeedDPS, LargeMotor, OUTPUT_B, OUTPUT_C


motor_right = LargeMotor(OUTPUT_C)
motor_right.reset()


# motor_right.speed_sp = SpeedDPS(200)
motor_right.speed_sp = 200
motor_right.run_forever()
# rotates the motor at 200 RPM (rotations-per-minute) for five seconds.
# LargeMotor.on_for_seconds(SpeedDPS(200), 5)

print("running ...")
for i in range(0, 10):

    print("Speed = ", motor_right.speed)
    time.sleep(1)

print("Speed_i", motor_right._speed_i)
print("Stop")
motor_right.stop()