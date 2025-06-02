#!/usr/bin/env python3


# Compatible with server_multi_test2.py from PC file

import time 
from messaging import *

# from pybricks import ev3brick as brick
# from pybricks.hubs import EV3Brick
# from pybricks.ev3devices import Motor
# from pybricks.parameters import Port

# from pybricks.messaging import BluetoothMailboxClient, TextMailbox
# import pybricks.messaging as pm
# pm.EV3_RFCOMM_CHANNEL=5

# from utils_fnct import interpreter

# Initialize the EV3 Brick.
# ev3 = EV3Brick()


# This is the name of the remote EV3 or PC we are connecting to.
# SERVER = "camilo-pc"
SERVER = "40:23:43:76:A9:D2"
client = BluetoothMailboxClient()

print("establishing connection...")
client.connect(SERVER)


mbox = TextMailbox("ev2", client)
print("connected!")


mbox.wait()
message_pc = mbox.read()
print(message_pc)                              #  Server
message = "hello PC!"
mbox.send(message)                              # Response
# ev3.screen.draw_text(10, 50, message_pc)