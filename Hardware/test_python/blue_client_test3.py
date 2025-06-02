#!/usr/bin/env pybricks-micropython

#
# Bluetooth Notes:
# ------------------------------------------------------------------------------
# Before running this program, make sure the client and server EV3 bricks are
# paired using Bluetooth, but do NOT connect them. The program will take care
# of establishing the connection.

# The server must be started before the client!

# ------------------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------------------
T_wait = 0                                  # time.sleep per cicly

# ------------------------------------------------------------------------------

import time
from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port

from pybricks.messaging import BluetoothMailboxClient, TextMailbox
import pybricks.messaging as pm
pm.EV3_RFCOMM_CHANNEL=5

from utils_fnct import interpreter


# Motor settings
motor_right = Motor(Port.C)
motor_left = Motor(Port.B)
# motor_left.hold()

# This is the name of the remote EV3 or PC we are connecting to.
# SERVER = "camilo-pc"
SERVER = "40:23:43:76:A9:D2"

client = BluetoothMailboxClient()
# time.sleep(1)

print("establishing connection...")
client.connect(SERVER)
# time.sleep(1)

mbox = TextMailbox("greeting", client)
print("connected!")

# In this program, the client sends the first message and then waits for the
# server to reply.
mbox.wait()
print(mbox.read())          # Server
mbox.send("hello PC!")      # Response


# Init. Motors
speed_right = 50                # Max. Speed 1000 deg/s
motor_right.run(speed_right)
speed_left = 50                 
motor_left.run(speed_left)


while True :

    # Wait for State Request     
    mbox.wait()    
    print(mbox.read())

    # Send State
    spd_right = str(motor_right.speed())
    spd_left = str(motor_left.speed())
    mbox.send(spd_right+","+spd_left)

    # Wait for Policy val.
    mbox.wait()
    speed_right, speed_left, stop = interpreter.policy_interpreter(mbox.read())
    
    if not stop :
        motor_right.run(speed_right)
        motor_left.run(speed_left)

        print("Right Speed = ", speed_right, motor_right.speed())
        print("Left Speed = ", speed_left, motor_left.speed())
    else:
        motor_right.brake()
        motor_left.brake()
        # print("STOP")        
       
    # ACK
    mbox.send("ACK")

    # ---- PROGRAM ---- #



    time.sleep(T_wait)


