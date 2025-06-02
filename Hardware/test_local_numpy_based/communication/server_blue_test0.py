#!/usr/bin/env python3

import time 
from messaging import *



# Communication
server = BluetoothMailboxServer()
# mbox = TextMailbox("greeting", server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# Prepare the messages box
mbox = TextMailbox("greeting", server)

time.sleep(1)
mbox.send("SERVER ready to receive")
print("Now time to read")
print()

mbox.wait()
print(mbox.read())

flag = True
while(flag):
    data = mbox.read()
    print("data received = ", data)
    if data == "Ok":
        flag = False

    time.sleep(0.5)


print("END")


