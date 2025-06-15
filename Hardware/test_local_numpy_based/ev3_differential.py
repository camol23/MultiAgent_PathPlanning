#!/usr/bin/env python3

import time
import numpy as np
import math

from ev3dev2.motor import OUTPUT_B, OUTPUT_C, MoveDifferential, SpeedRPM, SpeedDPS
from ev3dev2.wheel import EV3EducationSetTire
from ev3dev2.wheel import EV3Tire
from ev3dev2.sensor.lego import GyroSensor

from control import model
from utils_fnc import op_funct

# print("Intro size_w")
# size_w = int(input())

# print("Intro Rot_flag Val")
# on_change_rot_flag = int(input())
on_change_rot_flag = 0
print("on_change_rot_flag = ", on_change_rot_flag)

size_w = 87
print("size_w = ", size_w)


num_iter = 50
Ts = 0.01

# Wheel.__init__(self, 56, 28)
# mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3EducationSetTire, 105)

mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3Tire, size_w) # 90 with speed = 300


# Initialize the tank's gyro sensor
mdiff.gyro = GyroSensor()

# Calibrate the gyro to eliminate drift, and to initialize the current angle as 0
mdiff.gyro.calibrate()


# Enable odometry
mdiff.odometry_start(theta_degrees_start=0.0, x_pos_start=0.0, y_pos_start=0.0)
# mdiff.on_for_distance(SpeedRPM(40), 600)



# Use odometry to drive to specific coordinates
# mdiff.on_to_coordinates(SpeedRPM(40), 500, 0) # (speed, circle_radius, distance)
# mdiff.on_arc_right(SpeedRPM(40), 200, 2*3.1416*200) 
# mdiff.on_arc_right(SpeedDPS(300), 200, 2*math.pi*200) 


# Store
angle_list = []
mdiff.on(SpeedDPS(-300), SpeedDPS(300)) # (left, right)

first_flag = True
for i in range(0, num_iter):

    angle = mdiff.gyro.angle
    print("Sensor angle = ", angle)
    print("mdiff.theta deg", op_funct.radToDeg(mdiff.theta) )    
    
    gyro_angle = model.wrapped_angle_360_deg_neg(-angle)      
    gyro_angle_rad = op_funct.degToRad(gyro_angle)    
    print("Converted angle = ", gyro_angle, gyro_angle_rad)

    print("Speed (l, r) = ", mdiff.left_motor.speed, mdiff.right_motor.speed)
    print()
    
    angle_list.append( angle )             # degrees/second

    if (i >= num_iter/2) and (first_flag) and (on_change_rot_flag) :
        first_flag = False 
        mdiff.stop()
        print()
        print("Change rotation ........ ")
        time.sleep(1)
        mdiff.on(SpeedDPS(300), SpeedDPS(-300))

    time.sleep(Ts)


print("Stop")
mdiff.stop(brake=True)

print("End position = ", mdiff.x_pos_mm, mdiff.y_pos_mm, mdiff.theta)
# Disable odometry
mdiff.odometry_stop()