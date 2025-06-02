#!/usr/bin/env python3

import time 
from messaging import *


# Check messaging file to update BDADDR_ANY

server = BluetoothMailboxServer()
# mbox = TextMailbox("greeting", server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# Prepare the messages box
mbox_ev1 = TextMailbox("ev2", server)
 
# In this program, the server waits for the client to send the first message
# and then sends a reply.

time.sleep(2)
mbox_ev1.send("EV3_2 to PC client: Ready")
print("Now time to read")
print()