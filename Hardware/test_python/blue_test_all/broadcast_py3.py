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
class BLEAdvertisement:
    def __init__(self, board_id):
        self.board_id = board_id
        self.bus = None
        self.adapter = None
        self.mainloop = None
        self.ad_manager = None
        self.advertisement = None
        self.running = False
        self.path = "/org/bluez/ev3_adv"
        
    def setup(self):
        # Initialize D-Bus
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()
        self.mainloop = GLib.MainLoop()
        
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
    
    def create_advertisement(self, message):
        # Create a dict for the advertisement
        ad_props = {
            "Type": "peripheral",
            "ServiceUUIDs": ["180d"],  # Heart Rate Service UUID
            "LocalName": "EV3-"+str(self.board_id),
            "ManufacturerData": {
                0xFFFF: dbus.Array([
                    dbus.Byte(self.board_id),  # Board ID
                    *[dbus.Byte(c) for c in message.encode()]  # Actual message
                ])
            }
        }
        
        # Register the advertisement
        if self.advertisement:
            try:
                self.ad_manager.UnregisterAdvertisement(self.path)
            except:
                pass
        
        # self.advertisement = dbus.service.Object(self.bus, self.path)
        self.advertisement = dbus_service.Object(self.bus, self.path)
        
        # Add methods to the advertisement object
        def get_properties():
            return ad_props
        
        def get_path():
            return self.path
            
        def release():
            print("Advertisement released")
            
        self.advertisement.Get = lambda interface, prop: ad_props.get(prop)
        self.advertisement.GetAll = lambda interface: ad_props
        self.advertisement.GetPath = get_path
        self.advertisement.Release = release
        
        self.advertisement._dbus_interface = LE_ADVERTISEMENT_INTERFACE
        
        # Register the advertisement
        self.ad_manager.RegisterAdvertisement(
            self.advertisement.GetPath(),
            {},
            reply_handler=self.register_ad_cb,
            error_handler=self.register_ad_error_cb
        )
        
        return True
    
    def register_ad_cb(self):
        print("Advertisement registered")
    
    def register_ad_error_cb(self, error):
        # print(f"Failed to register advertisement: {error}")
        print("Failed to register advertisement:",error)
        self.mainloop.quit()
    
    def start_advertising(self, message):
        if not self.create_advertisement(message):
            return False
        
        # Start the main loop
        self.running = True
        self.mainloop_thread = threading.Thread(target=self.run_mainloop)
        self.mainloop_thread.daemon = True
        self.mainloop_thread.start()
        
        return True
    
    def run_mainloop(self):
        self.mainloop.run()
    
    def stop_advertising(self):
        if self.advertisement:
            try:
                self.ad_manager.UnregisterAdvertisement(self.path)
            except:
                pass
                
        self.running = False
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()

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
    # print(f"Starting EV3 BLE Node {BOARD_ID}")
    print("Starting EV3 BLE Node", BOARD_ID)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("This script must be run as root. Try using sudo.")
        sys.exit(1)
    
    # Initialize BLE advertisement
    advertiser = BLEAdvertisement(BOARD_ID)
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
            # message = f"T:{temp:.1f},L:{light:.1f}"
            message = "T: 3"
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