#!/usr/bin/env python3
from pybricks.messaging import BluetoothMailboxClient, TextMailbox

# This demo makes your PC talk to an EV3 over Bluetooth.
#
# This is identical to the EV3 client example in ../bluetooth_client
#
# The only difference is that it runs in Python3 on your computer, thanks to
# the Python3 implementation of the messaging module that is included here.
# As far as the EV3 is concerned, it thinks it just talks to an EV3 client.
#
# So, the EV3 server example needs no further modifications. The connection
# procedure is also the same as documented in the messaging module docs:
# https://docs.pybricks.com/en/latest/messaging.html
#
# So, turn Bluetooth on on your PC and the EV3. You may need to make Bluetooth
# visible on the EV3. You can skip pairing if you already know the EV3 address.

# This is the address of the server EV3 we are connecting to.
import time
import pybricks.messaging as pm
pm.EV3_RFCOMM_CHANNEL=5


#SERVER = "ticket"
SERVER = "F0:45:DA:D2:C8:88"

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


# Errors solved like:
# https://github.com/orgs/pybricks/discussions/1554
# https://github.com/orgs/pybricks/discussions/1193
#https://github.com/pybricks/support/issues/902#issuecomment-1374157195