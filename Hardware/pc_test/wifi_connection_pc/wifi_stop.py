#!/usr/bin/env python3

import rpyc
import time
# Create a RPyC connection to the remote ev3dev device.
# Use the hostname or IP address of the ev3dev device.
# If this fails, verify your IP connectivty via ``ping X.X.X.X``
conn_1 = rpyc.classic.connect('10.16.12.26')
conn_2 = rpyc.classic.connect('10.16.12.25')

# import ev3dev2 on the remote ev3dev device
ev3dev2_1_motor = conn_1.modules['ev3dev2.motor']
ev3dev2_2_motor = conn_2.modules['ev3dev2.motor']

ev3dev2_1_motor_B = conn_1.modules['ev3dev2.motor']
motor_left_ev1  = ev3dev2_1_motor_B.LargeMotor(ev3dev2_1_motor_B.OUTPUT_B)

ev3dev2_2_motor_B = conn_2.modules['ev3dev2.motor']
motor_left_ev2  = ev3dev2_2_motor_B.LargeMotor(ev3dev2_2_motor_B.OUTPUT_B)

# Use the LargeMotor and TouchSensor on the remote ev3dev device
motor_1 = ev3dev2_1_motor.LargeMotor(ev3dev2_1_motor.OUTPUT_C)
motor_2 = ev3dev2_2_motor.LargeMotor(ev3dev2_2_motor.OUTPUT_C)


motor_1.stop()
motor_2.stop()

motor_left_ev1.stop()
motor_left_ev2.stop()

