#!/usr/bin/env python3

import time 
from messaging import *



SERVER = "F0:45:DA:11:92:74"  # One

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
print(mbox.read())              # Server
mbox.send("hello SERVER!")      # Response

print("Sleeping time ...")
time.sleep(5)
print("Send Ok ...")
mbox.send("Ok")
