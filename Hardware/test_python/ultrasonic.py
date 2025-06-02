#!/usr/bin/env pybricks-micropython


from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, UltrasonicSensor
from pybricks.parameters import Port

import time 

# Initialize the EV3 Brick.
ev3 = EV3Brick()


# Initialize the Ultrasonic Sensor. It is used to detect
# obstacles as the robot drives around.
obstacle_sensor = UltrasonicSensor(Port.S2)

while True :
    print(obstacle_sensor.distance())

    time.sleep(1)

