import sys
import socket
# import bluetooth

# Print available socket families
print("Available Socket Families:")
for attr in dir(socket):
    if attr.startswith('AF_'):
        print(attr)

# Print Bluetooth library information
# print("\nBluetooth Library Info:")
# print(bluetooth.__file__)