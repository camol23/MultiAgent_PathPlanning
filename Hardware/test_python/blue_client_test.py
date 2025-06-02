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
# time.sleep(1)

print("establishing connection...")
client.connect(SERVER)
time.sleep(1)

mbox = TextMailbox("greeting", client)
state_box = TextMailbox("state", client)
print("connected!")

# In this program, the client sends the first message and then waits for the
# server to reply.
mbox.wait()
print(mbox.read())          # Server
mbox.send("hello PC!")      # Response

while True :

    # Wait for State Request     
    mbox.wait()    
    print(mbox.read())

    # Send State
    # mbox.send("STATE")
    state_box.send("STATE")

    # Wait for Policy val.
    mbox.wait()    
    print(mbox.read())
       
    # ACK
    mbox.send("ACK")

    # ---- PROGRAM ---- #



    time.sleep(1)


