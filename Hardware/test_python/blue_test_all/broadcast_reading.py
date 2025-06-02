#!/usr/bin/env python3

import bluetooth
import binascii
import ubinascii
from micropython import const

_IRQ_SCAN_RESULT = const(5)
_IRQ_SCAN_DONE = const(6)

def bt_irq(event, data):
    if event == _IRQ_SCAN_RESULT:
        # addr_type, addr, adv_type, rssi, adv_data = data
        addr_type, addr, _, rssi, adv_data = data
        
        # Convert address to string
        addr_str = ":".join(["{:02x}".format(b) for b in addr])
        
        # Look for our specific manufacturer ID (FFFF)
        manu_data = None
        i = 0
        while i < len(adv_data):
            length = adv_data[i]
            if i + length + 1 > len(adv_data):
                break
            
            adv_type = adv_data[i+1]
            if adv_type == 0xFF and length >= 3:  # Manufacturer Specific Data
                if adv_data[i+2] == 0xFF and adv_data[i+3] == 0xFF:  # Our ID: FFFF
                    manu_data = adv_data[i+4:i+length+1]
                    break
            i += length + 1
            
        if manu_data:
            # Convert bytes to string
            try:
                message = bytes(manu_data).decode()
                # print(f"Device: {addr_str}, RSSI: {rssi}dB, Message: {message}")
            except:
                # print(f"Device: {addr_str}, RSSI: {rssi}dB, Raw data: {ubinascii.hexlify(bytes(manu_data))}")
                print("execept")
    
    elif event == _IRQ_SCAN_DONE:
        print("Scan complete")
        # Restart scanning
        ble.gap_scan(0, 30000, 30000)

# Initialize BLE
ble = bluetooth.BLE()
ble.active(True)
ble.irq(bt_irq)

# Start scanning
ble.gap_scan(0, 30000, 30000)
print("Scanning...")