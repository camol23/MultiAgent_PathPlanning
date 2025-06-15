#!/usr/bin/env python3


from ev3dev2.sensor.lego import GyroSensor

from ev3dev2.sound import Sound

from time import sleep

# Connect gyro and touch sensors to any sensor ports

gy = GyroSensor() 

sound = Sound()

# Stop program by long-pressing touch sensor button

# while not ts.is_pressed:
for i in range(0, 10):

    angle = gy.angle

    print(str(angle) + ' degrees')

    # sound.play_tone(1000+angle*10, 1)

    sleep(0.5)