#!/usr/bin/env micropython

import bluetooth
import time
import struct
# import ubinascii
from micropython import const
import machine
import random

# Board configuration - CHANGE THIS FOR EACH BOARD
BOARD_ID = 1  # Set this to 1, 2, or 3 for different boards

# BLE constants
_IRQ_SCAN_RESULT = const(5)
_IRQ_SCAN_DONE = const(6)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATT_WRITE = const(3)

# Initialize BLE
ble = bluetooth.BLE()
ble.active(True)

# Simulated sensor for demo - replace with actual sensors
def read_sensor():
    # Simulate temperature and humidity reading
    temp = 20 + random.random() * 10
    humid = 50 + random.random() * 20
    return temp, humid

# Advertising payload builder
def advertise_sensor_data(board_id, message):
    # Convert message to bytes
    message_bytes = message.encode()
    
    # Build advertisement payload
    payload = bytearray()
    
    # Flags
    payload.extend(struct.pack('BB', 2, 0x01))  # Length, type
    payload.extend(b'\x06')  # LE General Discoverable + BR/EDR Not Supported
    
    # Complete local name
    # name = f"NODE-{board_id}"
    name = "NODE-"+str(board_id)
    name_bytes = name.encode()
    payload.extend(struct.pack('BB', len(name_bytes) + 1, 0x09))  # Length, type
    payload.extend(name_bytes)
    
    # Manufacturer specific data with our ID and message
    manuf_data = bytearray()
    manuf_data.extend(b'\xFF\xFF')  # Our manufacturer ID (0xFFFF)
    manuf_data.append(board_id)  # Board ID as first byte
    manuf_data.extend(message_bytes)  # Actual message
    
    payload.extend(struct.pack('BB', len(manuf_data) + 1, 0xFF))  # Length, type
    payload.extend(manuf_data)
    
    return payload

# Callback for BLE events
def bt_irq(event, data):
    if event == _IRQ_SCAN_RESULT:
        # addr_type, addr, adv_type, rssi, adv_data = data
        addr_type, addr, _, rssi, adv_data = data
        
        # Process advertisement data
        process_adv_data(addr, addr_type, rssi, adv_data)
    
    elif event == _IRQ_SCAN_DONE:
        # Restart scanning
        ble.gap_scan(30000, 30000, 30000)

# Process advertisement data looking for PC messages
def process_adv_data(addr, addr_type, rssi, adv_data):
    # Look for manufacturer-specific data
    target_id = 0
    message = None
    
    i = 0
    while i < len(adv_data):
        if i + 1 >= len(adv_data):
            break
            
        length = adv_data[i]
        if i + length + 1 > len(adv_data):
            break
        
        adv_type = adv_data[i+1]
        
        # Check for manufacturer specific data
        if adv_type == 0xFF and length >= 5:  # Type + ManufID(2) + TargetID(1) + Data(1+)
            # Check our manufacturer ID (FFFF)
            if adv_data[i+2] == 0xFF and adv_data[i+3] == 0xFF:
                # Extract target ID
                target_id = adv_data[i+4]
                
                # Extract message if target is 0 (broadcast) or our board_id
                if target_id == 0 or target_id == BOARD_ID:
                    message_bytes = adv_data[i+5:i+length+1]
                    
                    message = bytes(message_bytes).decode()
                    # try:
                    #     message = bytes(message_bytes).decode()
                    # # except:
                    # #     message = ubinascii.hexlify(bytes(message_bytes)).decode()
        
        i += length + 1
    
    # Process message if it's for us
    if message and (target_id == 0 or target_id == BOARD_ID):
        process_message(message)

# Process messages from PC
def process_message(message):
    # print(f"Message from PC: {message}")
    print("Message from PC: "+str(message))
    # Add your command processing logic here
    # Example: if message.startswith("LED:ON"): turn_on_led()

# Start advertising our data
def start_advertising():
    while True:
        try:
            # Read sensor data
            temp, humid = read_sensor()
            
            # Create message
            # message = f"T:{temp:.1f},H:{humid:.1f}"
            message = "DATA"
            
            # Create advertisement data
            payload = advertise_sensor_data(BOARD_ID, message)
            
            # Set advertisement data
            ble.gap_advertise(100, adv_data=payload)
            # print(f"Advertising: {message}")
            print("Advertising: "+str(message))
            
            # Scan briefly for incoming messages
            ble.gap_scan(1000, 30000, 30000, True)
            
            # Wait a bit before next advertisement
            time.sleep(2)
            
        except Exception as e:
            # print(f"Error: {e}")
            print("Error: "+str(e))
            time.sleep(5)  # Wait before retry

# Configure BLE
# ble.config(gap_name=f"NODE-{BOARD_ID}")
ble.config(gap_name="NODE-"+str(BOARD_ID))
ble.irq(bt_irq)

# Start the main loop
# print(f"Starting BLE Node {BOARD_ID}")
print("Starting BLE Node "+str(BOARD_ID))
start_advertising()