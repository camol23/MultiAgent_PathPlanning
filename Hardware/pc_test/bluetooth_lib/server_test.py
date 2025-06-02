#!/usr/bin/env python3

import time 
from messaging import *



server = BluetoothMailboxServer()
# mbox = TextMailbox("greeting", server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# Prepare the messages box
mbox = TextMailbox("greeting", server)
state_box = TextMailbox("state", server)

# In this program, the server waits for the client to send the first message
# and then sends a reply.

time.sleep(1)
mbox.send("PC ready to receive")
print("Now time to read")
print()

mbox.wait()
print(mbox.read())

while True :    

    # Ask for data
    mbox.send("PC")

    # Receive State
    # mbox.wait()
    state_box.wait()      
    print(state_box.read())

    # COMPUTE: Policy 
    mbox.send("data")


    # Wait for ACK
    mbox.wait()             
    print(mbox.read())

    # --- next_ev3 --- #

    time.sleep(3)