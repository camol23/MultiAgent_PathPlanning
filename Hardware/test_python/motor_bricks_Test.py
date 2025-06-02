#!/usr/bin/env pybricks-micropython

from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Button
from pybricks.tools import print, wait

motor = Motor(Port.B)
speed = 100
motor.run(speed)

while True:
    bts = brick.buttons()
    if Button.LEFT in bts:
        speed = max(-1000, speed-100)

    elif Button.RIGHT in bts:
        speed = min(1000, speed+100)

    elif Button.CENTER in bts:
        break

    motor.run(speed)
    print(speed, motor.speed(), motor.angle())
    print(motor.control.pid() )
    wait(100)