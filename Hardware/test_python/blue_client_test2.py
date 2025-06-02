#!/usr/bin/env pybricks-micropython

# Before running this program, make sure the client and server EV3 bricks are
# paired using Bluetooth, but do NOT connect them. The program will take care
# of establishing the connection.

# The server must be started before the client!

import time
from pybricks import ev3brick as brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port

from pybricks.messaging import BluetoothMailboxClient, TextMailbox

import pybricks.messaging as pm
pm.EV3_RFCOMM_CHANNEL=5


def policy_interpreter(data):
    data = data.split(",")

    data_right = min(400, int(data[0]) )
    data_left = min(400, int(data[1]) )

    return data_right, data_left



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
    speed_right, speed_left = policy_interpreter(mbox.read())
    print("Right Speed = ", speed_right, motor_right.speed())
    print("Left Speed = ", speed_left, motor_left.speed())
    motor_right.run(speed_right)
    motor_left.run(speed_left)
       
    # ACK
    mbox.send("ACK")

    # ---- PROGRAM ---- #



    time.sleep(1)


