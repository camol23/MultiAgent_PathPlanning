#!/usr/bin/env python3
import subprocess
import time
import binascii
import threading
import dbus
import dbus.service as dbus_service
import dbus.mainloop.glib
from gi.repository import GLib
import sys
import random
import os

import dbus.exceptions
# Board configuration - CHANGE THIS FOR EACH BOARD
BOARD_ID = 1  # Set this to 1, 2, or 3 for different boards

# BLE Service UUIDs and characteristics
BLUEZ_SERVICE_NAME = "org.bluez"
ADAPTER_INTERFACE = "org.bluez.Adapter1"
DEVICE_INTERFACE = "org.bluez.Device1"
GATT_MANAGER_INTERFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_INTERFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_INTERFACE = "org.bluez.LEAdvertisement1"

# Class for handling BLE advertisements
class BLEAdvertisement(dbus.service.Object):
    def __init__(self, bus, board_id, path="/org/bluez/ev3_adv"):
        self.board_id = board_id
        self.path = path
        self.ad_props = {}
        self.bus = bus
        self.adapter = None
        self.mainloop = None
        self.ad_manager = None
        self.running = False
        dbus.service.Object.__init__(self, bus, self.path)
    
    def setup(self):
        # Get the adapter
        self.adapter = self.find_adapter()
        if not self.adapter:
            print("Bluetooth adapter not found")
            return False
        
        # Get the LE advertising manager
        self.ad_manager = self.get_ad_manager()
        if not self.ad_manager:
            print("LEAdvertisingManager1 interface not found")
            return False
            
        print("BLE Advertisement setup complete")
        return True
    
    def find_adapter(self):
        remote_om = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, "/"),
                                  "org.freedesktop.DBus.ObjectManager")
        objects = remote_om.GetManagedObjects()
        
        for obj_path, interfaces in objects.items():
            if ADAPTER_INTERFACE in interfaces:
                return obj_path
                
        return None
    
    def get_ad_manager(self):
        return dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter),
                             LE_ADVERTISING_MANAGER_INTERFACE)
    
    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise InvalidArgsException()
        return self.ad_props
    
    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise InvalidArgsException()
        return self.ad_props.get(prop)
    
    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature="ssv")
    def Set(self, interface, prop, value):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise InvalidArgsException()
        self.ad_props[prop] = value
    
    @dbus.service.method(LE_ADVERTISEMENT_INTERFACE)
    def Release(self):
        print("Advertisement released")
        
    def register_ad_cb(self):
        print("Advertisement registered")
    
    def register_ad_error_cb(self, error):
        print("Failed to register advertisement:", error)
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()
            
    def start_advertising(self, message):
        # Set up advertisement properties
        self.ad_props = {
            "Type": dbus.String("peripheral"),
            "ServiceUUIDs": dbus.Array(["180d"], signature="s"),  # Heart Rate Service UUID
            "LocalName": dbus.String("EV3-" + str(self.board_id)),
            "ManufacturerData": dbus.Dictionary({
                dbus.UInt16(0xFFFF): dbus.Array([
                    dbus.Byte(self.board_id),  # Board ID
                    *[dbus.Byte(c) for c in message.encode()]  # Actual message
                ], signature=dbus.Signature("y"))
            }, signature=dbus.Signature("qv"))
        }
        
        # Register the advertisement
        self.ad_manager.RegisterAdvertisement(
            self.path,
            dbus.Dictionary({}, signature="sv"),
            reply_handler=self.register_ad_cb,
            error_handler=self.register_ad_error_cb
        )
        
        # Start the main loop if not already running
        if not self.running:
            self.running = True
            self.mainloop_thread = threading.Thread(target=self.run_mainloop)
            self.mainloop_thread.daemon = True
            self.mainloop_thread.start()
        
        return True
    
    def run_mainloop(self):
        self.mainloop.run()
    
    def stop_advertising(self):
        try:
            self.ad_manager.UnregisterAdvertisement(self.path)
            print("Advertisement unregistered")
        except Exception as e:
            print("Error unregistering advertisement:", e)
                
        self.running = False
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


# BLE Scanner class for receiving messages
class BLEScanner:
    def __init__(self, board_id):
        self.board_id = board_id
        self.running = False
        self.scan_process = None
    
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        print("BLE scanner started")
    
    def _scan_loop(self):
        try:
            while self.running:
                # Use hcitool to scan for advertisements
                self.scan_process = subprocess.Popen(
                    ["hcitool", "lescan", "--duplicates"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Start a separate thread to read from the process
                read_thread = threading.Thread(target=self._read_scan_output)
                read_thread.daemon = True
                read_thread.start()
                
                # Wait before restarting scan
                time.sleep(10)
                
                # Kill the process
                if self.scan_process:
                    self.scan_process.terminate()
                    self.scan_process = None
                
        except Exception as e:
            # print(f"Scan error: {e}")
            print("Scan error:"+ str(e))
        finally:
            self.running = False
    
    def _read_scan_output(self):
        try:
            while self.scan_process and self.running:
                line = self.scan_process.stdout.readline().decode('utf-8').strip()
                if line:
                    # Process the scan output
                    self._process_scan_line(line)
        except Exception as e:
            # print(f"Error reading scan output: {e}")
            print("Error reading scan output:"+ str(e))
    
    def _process_scan_line(self, line):
        parts = line.split(' ')
        if len(parts) >= 2:
            mac_address = parts[0]
            name = ' '.join(parts[1:])
            
            # Check if this is from our PC broadcaster
            if "PC-Master" in name or "PC-Broadcaster" in name:
                # Run hcidump to get advertisement data
                self._get_adv_data(mac_address)
    
    def _get_adv_data(self, mac_address):
        try:
            # Use hcidump to get advertisement data - this is a simplified example
            # In a real implementation, you would want to use a proper BLE library
            cmd = ["hcidump", "--raw", "hci"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for a short time to collect data
            time.sleep(1)
            proc.terminate()
            
            # Process the output to find manufacturer data
            # This is highly simplified - real implementation would parse BLE packets properly
            output = proc.stdout.read().decode('utf-8')
            if "FFFF" in output:
                # Found manufacturer data with our ID
                # Extract and process the message
                # This is a placeholder - would need proper parsing
                # print(f"Received message from PC: {output}")
                print("Received message from PC:",output)
        except Exception as e:
            # print(f"Error getting advertisement data: {e}")
            print("Error getting advertisement data:", (e))

    
    def stop_scanning(self):
        self.running = False
        if self.scan_process:
            self.scan_process.terminate()
            self.scan_process = None

# Simulated sensor for demo - replace with actual EV3 sensors
def read_sensor():
    # Here you would use the ev3dev Python API to read actual sensors
    # For example:
    # import ev3dev2.sensor as sensors
    # from ev3dev2.sensor.lego import TouchSensor, ColorSensor
    # touch = TouchSensor()
    # is_pressed = touch.is_pressed
    
    # For now, just simulate some readings
    temp = 20 + random.random() * 10
    light = 30 + random.random() * 50
    return temp, light

def main():
    print("Starting EV3 BLE Node", BOARD_ID)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("This script must be run as root. Try using sudo.")
        sys.exit(1)
    
    # Initialize D-Bus
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    mainloop = GLib.MainLoop()
    
    # Initialize BLE advertisement
    advertiser = BLEAdvertisement(bus, BOARD_ID)
    advertiser.mainloop = mainloop  # Pass the mainloop to the advertiser
    if not advertiser.setup():
        print("Failed to set up BLE advertisement")
        sys.exit(1)
    
    # Initialize scanner
    scanner = BLEScanner(BOARD_ID)
    scanner.start_scanning()
    
    try:
        while True:
            # Read sensor data
            temp, light = read_sensor()
            
            # Create message
            message = "Test Message"
            print("Broadcasting:", message)
            
            # Start advertising
            advertiser.start_advertising(message)
            
            # Wait before updating advertisement
            time.sleep(5)
            
            # Stop advertising (to update with new data)
            advertiser.stop_advertising()
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        advertiser.stop_advertising()
        scanner.stop_scanning()
        print("BLE Node stopped")

if __name__ == "__main__":
    main()