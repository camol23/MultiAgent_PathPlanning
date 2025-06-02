#!/usr/bin/env micropython

from ev3dev2.motor import OUTPUT_B, OUTPUT_C, MoveDifferential, SpeedRPM
from ev3dev2.wheel import EV3EducationSetTire
from ev3dev2.sensor.lego import GyroSensor

# from pybricks import ev3brick as brick # Doesn't work

# Wheel.__init__(self, 56, 28)
mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3EducationSetTire, 105)

# Initialize the tank's gyro sensor
mdiff.gyro = GyroSensor()

# Calibrate the gyro to eliminate drift, and to initialize the current angle as 0
mdiff.gyro.calibrate()

# Rotate 90 degrees clockwise
# mdiff.turn_right(SpeedRPM(40), 90)

# Drive forward 500 mm
#mdiff.on_for_distance(SpeedRPM(40), 500)

# Enable odometry
mdiff.odometry_start(theta_degrees_start=0.0, x_pos_start=0.0, y_pos_start=0.0)
# mdiff.on_for_distance(SpeedRPM(40), 600)


# def on_to_coordinates(self, speed, x_target_mm, y_target_mm, brake=True, block=True):
#mdiff.on_to_coordinates(SpeedRPM(40), 835, -580, brake=True, block=True)


# Use odometry to drive to specific coordinates
# mdiff.on_to_coordinates(SpeedRPM(40), 500, 0)
mdiff.on_arc_right(SpeedRPM(40), 200, 2*3.1416*200) 


print("End position = ", mdiff.x_pos_mm, mdiff.y_pos_mm, mdiff.theta)
# Disable odometry
mdiff.odometry_stop()