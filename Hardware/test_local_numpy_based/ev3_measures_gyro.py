#!/usr/bin/env python3

import time
import numpy as np
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor.lego import GyroSensor

from utils_fnc import op_funct

num_iter = 10
Ts = 0.01

gyro = GyroSensor() 
# gyro.calibrate()                    
# gyro.reset()

angle_list = []
rate_list = []

# Motor
motor_right = LargeMotor(OUTPUT_C)
motor_left = LargeMotor(OUTPUT_B)
motor_right.reset()
motor_left.reset()
motor_right.stop()
motor_left.stop()
# motor_right._speed_p = 400
# motor_right._speed_i = 1200
# motor_right._speed_d = 5
# motor_left._speed_p = 400
# motor_left._speed_i = 1200
# motor_left._speed_d = 5
print("Speed PID", motor_right._speed_p, motor_right._speed_i, motor_right._speed_d)

print("running ...")
motor_left.speed_sp = -200
motor_left.run_forever()
motor_right.speed_sp = 200
motor_right.run_forever()
for i in range(0, num_iter):

    print("angle_and_rate", gyro.angle_and_rate)
    print("circle_angle()", gyro.circle_angle())
    print(gyro.rate, gyro.angle)
    rate_list.append( gyro.rate )             # degrees/second

    time.sleep(Ts)


print("Stop")
motor_right.stop()
motor_left.stop()

rate_np = np.array(rate_list)
angle_np = rate_np*Ts
rate_mean = np.mean(rate_np)
rate_std = np.std(rate_np)
print("Rate range [d/s] ", np.min(rate_np), np.max(rate_np))
print("Rate mean ", rate_mean)
print("Rate STD ", rate_std)
print("Angle from Rate mean ", np.mean(angle_np) )
print("Angle from Rate STD ", np.std(angle_np) )
print()

rate_list_rad = [op_funct.degToRad(deg_val) for deg_val in rate_list]
print(" rad/s ")
print("Vel mean ", np.mean(rate_list_rad))
print("Vel STD ", np.std(rate_list_rad))
print("Vel Var. ", np.var(rate_list_rad))
print()

r_width = 0.053 # m -> robot center to wheel center
rate_list_m = [op_funct.deg2m(deg_val, r_width) for deg_val in rate_list]
print(" m/s ")
print("Vel mean ", np.mean(rate_list_m))
print("Vel STD ", np.std(rate_list_m))
print("Vel Var. ", np.var(rate_list_m))
print()


# Angle test
# angle_ref = 30
# counter = 0 

# print("running ...")
# motor_right.speed_sp = 200
# motor_right.run_forever()
# while(counter < num_iter):

#     angle = gyro.angle
#     if angle < angle_ref :
#         angle_list.append(angle)
#         angle.reset()

#         conuter = conuter + 1

#     time.sleep(Ts)


# print("Stop")
# motor_right.stop()
# motor_left.stop()

# print("Angle range [d/s] ", np.min(angle_list), np.max(angle_list))
# print("Angle mean ", np.mean(angle_list))
# print("Angle STD ", np.std(angle_list))
