
#!/usr/bin/env python3

'''
    Motor drives for Duty Cycle
'''

import time
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C


motor_right = LargeMotor(OUTPUT_C)
motor_left = LargeMotor(OUTPUT_B)
motor_right.reset()
motor_left.reset()



# motor_right.speed_sp = 200
# motor_right.run_forever()
motor_right.duty_cycle_sp = 50  # ~> 450 deg/s
motor_right.run_direct()

motor_left.duty_cycle_sp = 20 # ~> 160 deg/s
motor_left.run_direct()

print("running ...")
for i in range(0, 10):

    print("Speed = ", motor_right.speed, motor_left.speed)    
    time.sleep(1)

print("Speed PID", motor_right._speed_p, motor_right._speed_i, motor_right._speed_d)
print("Stop")
motor_right.stop()
motor_left.stop()