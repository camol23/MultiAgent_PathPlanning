#!/usr/bin/env python3

# from ev3dev2.sensor import Sensor
# from ev3dev2.bluetooth import BluetoothDevice

# import time

# def ev3dev2_bluetooth_example():
#     """
#     Bluetooth communication using python-ev3dev2
#     Recommended modern approach for LEGO Mindstorms EV3
#     """
#     # Discover and connect to Bluetooth devices
#     device = BluetoothDevice()
#     device.connect()  # Connect to paired device
    
#     # Send and receive data
#     while 1 :
#         device.send("Hello EV3")
#         time.sleep(0.3)

#     # received_data = device.receive()


import ev3

def python_ev3_bluetooth():
    """
    Bluetooth communication using python-ev3 library
    Alternative approach for EV3 devices
    """
    # Discover Bluetooth devices
    devices = ev3.list_bluetooth_devices()
    
    # Connect to a specific device
    connection = ev3.connect_bluetooth(device_address)
    connection.send("Data")