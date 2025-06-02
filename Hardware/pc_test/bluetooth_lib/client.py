#!/usr/bin/env python3
#from pybricks.messaging import BluetoothMailboxServer, TextMailbox
# from messaging import BluetoothMailboxServer, TextMailbox
from messaging import *
import time 


SERVER = "ticket"
# SERVER = "F0:45:DA:D2:C8:88"

# SERVER = "00:16:53:61:CA:75"
# SERVER = "F0:45:DA:11:92:74"

client = BluetoothMailboxClient()
# mbox = TextMailbox("greeting", client)
time.sleep(1)
print("establishing connection...")
client.connect(SERVER)
time.sleep(1)
print("connected!")

mbox = TextMailbox("greeting", client)

# In this program, the client sends the first message and then waits for the
# server to reply.
# mbox.send("hello!")
# mbox.wait()
# print(mbox.read())

while(1):
    mbox.wait()
    speed_msg='Speed: '+str(mbox.read())+'\n'
    print(speed_msg)
    time.sleep(1)
    mbox.send("hello!")