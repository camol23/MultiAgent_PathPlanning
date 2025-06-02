#!/usr/bin/env pybricks-micropython

# Before running this program, make sure the client and server EV3 bricks are
# paired using Bluetooth, but do NOT connect them. The program will take care
# of establishing the connection.

# The server must be started before the client!

import time
from pybricks.messaging import BluetoothMailboxClient, TextMailbox

import pybricks.messaging as pm
pm.EV3_RFCOMM_CHANNEL=5

# This is the name of the remote EV3 or PC we are connecting to.
# SERVER = "camilo-pc"
SERVER = "40:23:43:76:A9:D2"

client = BluetoothMailboxClient()
time.sleep(1)

print("establishing connection...")
client.connect(SERVER)
time.sleep(1)

mbox = TextMailbox("greeting", client)
# print("establishing connection...")
# client.connect(SERVER)
print("connected!")

# In this program, the client sends the first message and then waits for the
# server to reply.
mbox.wait()
print(mbox.read())          # Server
mbox.send("hello PC!")      # Response

while True :     
    mbox.wait()    
    print(mbox.read())

    mbox.send("I'm connected .... ")      
    time.sleep(1)


# speed_msg='Speed: '+str(mbox.read())+'\n'

# Error
#   File "pybricks/messaging.py", line 318, in connect
#   File "pybricks/bluetooth.py", line 168, in handle_request
# OSError: 112


# while True :
#     mbox.send("hello PC!")
#     print("Waiting ...")
#     mbox.wait()
#     print(mbox.read())

#     print("I've Send ... ")
#     time.sleep(2)


# Traceback (most recent call last):
#   File "/home/robot/test_python/bluetooth_client.py", line 20, in <module>
#   File "pybricks/messaging.py", line 321, in connect
#   File "pybricks/messaging.py", line 318, in connect
#   File "pybricks/bluetooth.py", line 168, in handle_request
# OSError: [Errno 111] ECONNREFUSED
# Pybricks MicroPython v1.11 on 2020-05-06; linux version
# Use Ctrl-D to exit, Ctrl-E for paste mode