
#from pybricks.messaging import BluetoothMailboxServer, TextMailbox
# from messaging import BluetoothMailboxServer, TextMailbox
from messaging import *
import time 



server = BluetoothMailboxServer()
# mbox = TextMailbox("greeting", server)
mbox = TextMailbox(server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# In this program, the server waits for the client to send the first message
# and then sends a reply.

# time.sleep(15)
# print("Now time to read")
# mbox.wait()
# print(mbox.read())
# mbox.send("hello to EV3!")