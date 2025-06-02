#!/usr/bin/env python3

from ev3dev2.sensor import INPUT_2
from ev3dev2.sensor.lego import UltrasonicSensor

from ev3dev2.sound import Sound

import time

ultrasonic_sensor = UltrasonicSensor(INPUT_2)


sound = Sound()
sound.speak('Ready')
for i in range(0, 5):

    dist = ultrasonic_sensor.distance_centimeters

    print(dist)
    time.sleep(0.5)

sound.speak('Pep')