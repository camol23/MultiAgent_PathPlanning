#!/usr/bin/env python3

import time
import numpy as np
import math
from ev3dev2.motor import OUTPUT_B, OUTPUT_C, MoveDifferential, SpeedRPM, SpeedDPS
from ev3dev2.wheel import EV3EducationSetTire
from ev3dev2.wheel import EV3Tire
from ev3dev2.sensor.lego import GyroSensor



print("Intro size_w")
size_w = int(input())
print("size_w = ", size_w)

# Wheel.__init__(self, 56, 28)
# mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3EducationSetTire, 105)

mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3Tire, size_w) # 86 with speed = 300


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
mdiff.on_arc_right(SpeedDPS(300), 200, 2*math.pi*200) 


print("End position = ", mdiff.x_pos_mm, mdiff.y_pos_mm, mdiff.theta)
# Disable odometry
mdiff.odometry_stop()