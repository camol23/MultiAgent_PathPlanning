#!/usr/bin/env python3

import time 
from messaging import *

# It can be connect with blue_multi_server_1_test_v2 One by One
# Not possible two connections bluetooth.error: (104, 'Connection reset by peer')

# This is the name of the remote EV3 or PC we are connecting to.
# SERVER = "40:23:43:76:A9:D2" # PC ADDR

SERVER = "F0:45:DA:11:92:74"
client = BluetoothMailboxClient()

print("establishing connection...")
client.connect(SERVER)


mbox = TextMailbox("ev1", client)
print("connected!")


mbox.wait()
message_pc = mbox.read()
print(message_pc)                              #  Server
message = "hello PC!"
mbox.send(message)                              # Response


client._clients[SERVER].close()


# Second EV3 
print()
print("Wait a second, actuallty  5s")
time.sleep(5)
print()
print("Second EV3 (ticket)")
SERVER_2 = "F0:45:DA:D2:C8:88"
client_2 = BluetoothMailboxClient()

print("establishing connection...")
client_2.connect(SERVER_2)
print("connected!")

mbox_2 = TextMailbox("ev2", client_2)



mbox_2.wait()
message_pc = mbox_2.read()
print(message_pc)                              #  Server
message = "hello PC!"
mbox_2.send(message)                              # Response


client_2._clients[SERVER_2].close()
