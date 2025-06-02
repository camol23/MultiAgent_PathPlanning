#!/usr/bin/env python3
#from pybricks.messaging import BluetoothMailboxServer, TextMailbox
# from messaging import BluetoothMailboxServer, TextMailbox
from messaging import *
import time 



server = BluetoothMailboxServer()
mbox = TextMailbox("greeting", server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# In this program, the server waits for the client to send the first message
# and then sends a reply.

time.sleep(1)
print("Now time to read")
mbox.wait()
print(mbox.read())
# mbox.send("hello to EV3!")




# Requirements:
# Update system packages
# sudo apt-get update

# # Install necessary dependencies
# sudo apt-get install -y \
#     bluetooth \
#     bluez \
#     libbluetooth-dev \
#     python3-dev \
#     build-essential \
#     python3-pip

# # Upgrade pip and setuptools
# pip3 install --upgrade pip

# # The most important:
# #pip3 install git+https://github.com/pybluez/pybluez.git

# # Errors:
# https://github.com/orgs/pybricks/discussions/1193
# https://github.com/pybricks/support/issues/902#issuecomment-1374157195

# # Git: commit give the code
# https://github.com/pybricks/pybricks-api/tree/master/examples/ev3/bluetooth_pc/pybricks

# # Official Docs
# https://pybricks.com/ev3-micropython/messaging.html

# # Alternative (Broadcasting)
# https://pybricks.com/project/micropython-ble-communication/