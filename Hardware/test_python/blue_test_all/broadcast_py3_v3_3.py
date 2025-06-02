#!/usr/bin/env python3
import subprocess
import time
import threading
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import sys
import random
import os
import shutil

# Board configuration - CHANGE THIS FOR EACH BOARD
BOARD_ID = 1  # Set this to 1, 2, or 3 for different boards

# BLE Service UUIDs and characteristics
BLUEZ_SERVICE_NAME = "org.bluez"
ADAPTER_INTERFACE = "org.bluez.Adapter1"
DEVICE_INTERFACE = "org.bluez.Device1"
GATT_MANAGER_INTERFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_INTERFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_INTERFACE = "org.bluez.LEAdvertisement1"

class Advertisement(dbus.service.Object):
    def __init__(self, bus, path, ad_props):
        self.path = path
        self.props = ad_props
        dbus.service.Object.__init__(self, bus, path)

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                "Interface " + interface + " not supported")
        return self.props

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                "Interface " + interface + " not supported")
        return self.props.get(prop, None)

    @dbus.service.method(LE_ADVERTISEMENT_INTERFACE)
    def Release(self):
        print("Advertisement released")

class BLEAdvertisement:
    def __init__(self, board_id):
        self.board_id = board_id
        self.bus = None
        self.adapter = None
        self.mainloop = None
        self.ad_manager = None
        self.advertisement = None
        self.running = False
        self.path = "/org/bluez/ev3_adv_" + str(board_id)
        self.supported = True  # Assume BLE advertising is supported initially
        
    def setup(self):
        # Initialize D-Bus
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()
        self.mainloop = GLib.MainLoop()
        
        # Get the adapter and check if it supports BLE advertising
        self.adapter = self.find_adapter()
        if not self.adapter:
            print("Bluetooth adapter not found")
            self.supported = False
            return False
        
        # Check if Bluetooth adapter supports LE advertising
        try:
            self.ad_manager = self.get_ad_manager()
            if not self.ad_manager:
                print("LEAdvertisingManager1 interface not found - advertising not supported")
                self.supported = False
                return False
        except dbus.exceptions.DBusException as e:
            print("Error accessing Bluetooth advertising: " + str(e))
            self.supported = False
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
        # If advertising not supported, don't try
        if not self.supported:
            return False
            
        # Limit message length to ensure it fits in advertisement (max ~20 chars)
        if len(message) > 20:
            message = message[:20]
            
        # Create a dict for the advertisement
        ad_props = {
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
        
        # Unregister any existing advertisement
        if self.advertisement:
            try:
                self.ad_manager.UnregisterAdvertisement(self.advertisement.path)
                self.advertisement.remove_from_connection()
            except Exception as e:
                print("Error unregistering previous advertisement: " + str(e))
        
        # Create and register the new advertisement
        try:
            self.advertisement = Advertisement(self.bus, self.path, ad_props)
            
            self.ad_manager.RegisterAdvertisement(
                self.path,
                dbus.Dictionary({}, signature="sv"),
                reply_handler=self.register_ad_cb,
                error_handler=self.register_ad_error_cb
            )
            return True
        except dbus.exceptions.DBusException as e:
            error_name = e.get_dbus_name()
            # If method doesn't exist, mark advertising as unsupported
            if "UnknownMethod" in error_name:
                print("BLE advertising not supported on this device")
                self.supported = False
            print("Failed to create advertisement: " + str(e))
            return False
        except Exception as e:
            print("Failed to create advertisement: " + str(e))
            return False
    
    def register_ad_cb(self):
        print("Advertisement registered")
    
    def register_ad_error_cb(self, error):
        # If this is the first error, mark advertising as unsupported
        self.supported = False
        print("Failed to register advertisement: " + str(error))
        # Don't quit the mainloop as we still want to receive
        # Instead log the error and continue
    
    def start_advertising(self, message):
        if not self.supported:
            # Alternative: Just print the message locally if we can't advertise
            print("Would advertise (if supported): " + message)
            return False
            
        if not self.create_advertisement(message):
            return False
        
        # Start the main loop if not already running
        if not self.running:
            self.running = True
            self.mainloop_thread = threading.Thread(target=self.run_mainloop)
            self.mainloop_thread.daemon = True
            self.mainloop_thread.start()
        
        return True
    
    def run_mainloop(self):
        try:
            self.mainloop.run()
        except Exception as e:
            print("Error in mainloop: " + str(e))
    
    def stop_advertising(self):
        if self.advertisement and self.supported:
            try:
                self.ad_manager.UnregisterAdvertisement(self.path)
                self.advertisement.remove_from_connection()
                self.advertisement = None
            except Exception as e:
                print("Error unregistering advertisement: " + str(e))
        
        self.running = False
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()

# Additional fallback mechanism for BLE scanning if dbus approach fails
class SimpleBLEDeviceScanner:
    def __init__(self, board_id):
        self.board_id = board_id
        self.running = False
        self.last_pc_message = "No messages"
        self.last_pc_seen = 0
        
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        print("Simple BLE scanner started")
    
    def _scan_loop(self):
        try:
            while self.running:
                # Use bluetoothctl to scan for devices
                cmd = "bluetoothctl scan on"
                try:
                    output = subprocess.check_output(cmd, shell=True, 
                                                    stderr=subprocess.STDOUT,
                                                    timeout=10)
                    lines = output.decode('utf-8', errors='replace').splitlines()
                    for line in lines:
                        if "PC-Master" in line:
                            print("Detected PC-Master device: " + line)
                except subprocess.SubprocessError as e:
                    print("Error scanning: " + str(e))
                
                time.sleep(5)  # Wait before scanning again
        except Exception as e:
            print("Scan error: " + str(e))
        finally:
            self.running = False
    
    def stop_scanning(self):
        self.running = False

class BLEScanner:
    def __init__(self, board_id):
        self.board_id = board_id
        self.running = False
        self.scan_process = None
        self.dump_process = None
        self.last_pc_message = "No messages"
        self.last_pc_seen = 0
        
        # Check if hcidump is available
        self.hcidump_available = shutil.which('hcidump') is not None
        if not self.hcidump_available:
            print("Warning: 'hcidump' not found. Will use alternative scanning method.")
        
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        
        # Start the scan thread
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
        # If hcidump is available, use it for monitoring
        if self.hcidump_available:
            self.monitor_thread = threading.Thread(target=self._monitor_advertisements)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        else:
            # Alternative method using btmon if available
            if shutil.which('btmon'):
                self.monitor_thread = threading.Thread(target=self._monitor_with_btmon)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
            else:
                print("Warning: Neither 'hcidump' nor 'btmon' found. Limited functionality.")
                # Simple scan only - no detailed packet capture
        
        print("BLE scanner started")
    
    def _scan_loop(self):
        try:
            while self.running:
                # Use hcitool to scan for advertisements
                cmd = ["hcitool", "lescan", "--duplicates"]
                
                # On some systems, it might be bluetoothctl instead
                if not shutil.which('hcitool'):
                    if shutil.which('bluetoothctl'):
                        cmd = ["bluetoothctl", "scan", "on"]
                    else:
                        print("Error: Neither hcitool nor bluetoothctl found!")
                        return
                
                try:
                    self.scan_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Process scan output
                    while self.running:
                        line = self.scan_process.stdout.readline().decode('utf-8', errors='replace').strip()
                        if line:
                            # If we have no packet sniffer, try to extract data from scan output
                            if not self.hcidump_available and not shutil.which('btmon'):
                                if "PC-Master" in line:
                                    print("Detected PC-Master: " + str(line))
                except Exception as e:
                    print("Scan process error: " + str(e))
                
                # Wait before restarting scan if process exits
                time.sleep(5)
                
        except Exception as e:
            print("Scan error: " + str(e))
        finally:
            if self.scan_process:
                try:
                    self.scan_process.terminate()
                except:
                    pass
                self.scan_process = None
            self.running = False
    
    def _monitor_advertisements(self):
        try:
            # Use hcidump to monitor raw advertisement data
            self.dump_process = subprocess.Popen(
                ["hcidump", "--raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            buffer = b""
            while self.running:
                # Read data from hcidump
                data = self.dump_process.stdout.read(1)
                if not data:
                    break
                    
                buffer += data
                
                # Check for complete packets
                if b"\n" in buffer:
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()  # Keep the incomplete line
                    
                    # Process complete lines
                    for line in lines:
                        self._process_adv_data(line.decode('utf-8', errors='replace').strip())
            
        except Exception as e:
            print("Error monitoring advertisements: " + str(e))
        finally:
            # Clean up
            if self.dump_process:
                try:
                    self.dump_process.terminate()
                except:
                    pass
                self.dump_process = None
    
    def _monitor_with_btmon(self):
        try:
            # Use btmon as an alternative to hcidump
            self.dump_process = subprocess.Popen(
                ["btmon", "-t"],  # Text output format
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Process btmon output
            while self.running:
                line = self.dump_process.stdout.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                    
                # Look for interesting data
                if "PC-Master" in line:
                    print("Detected PC advertisement: " + str(line))
                    # Extract data if possible
                    if "Data:" in line:
                        data_part = line.split("Data:")[1].strip()
                        self._process_btmon_data(data_part)
                        
        except Exception as e:
            print("Error with btmon: " + str(e))
        finally:
            if self.dump_process:
                try:
                    self.dump_process.terminate()
                except:
                    pass
                self.dump_process = None
    
    def _process_btmon_data(self, data):
        # Process data from btmon output
        # Format is typically hex values separated by spaces
        try:
            # Look for our manufacturer ID (0xFFFF)
            if "ff ff" in data.lower():
                parts = data.lower().split("ff ff")
                if len(parts) < 2:
                    return
                
                # Extract target board ID and message
                hex_values = parts[1].strip().split()
                if not hex_values:
                    return
                
                # First byte is target board ID
                target_id = int(hex_values[0], 16)
                
                # Check if message is for this board
                if target_id != 0 and target_id != self.board_id:
                    return  # Not for us
                
                # Convert remaining bytes to message
                try:
                    message_bytes = bytes([int(h, 16) for h in hex_values[1:]])
                    message = message_bytes.decode('ascii', errors='replace')
                    
                    # Update last message
                    self.last_pc_message = message
                    self.last_pc_seen = time.time()
                    
                    print("Received from PC (Target: " + str(target_id) + "): " + str(message))
                except Exception as e:
                    print("Error decoding message: " + str(e))
                    
        except Exception as e:
            print("Error processing btmon data: " + str(e))
    
    def _process_adv_data(self, line):
        # Looking for PC advertisements
        if "PC-Master" in line and "ff ff" in line.lower():
            try:
                # Extract raw hex data following 0xFF 0xFF (manufacturer specific data)
                parts = line.lower().split("ff ff")
                if len(parts) < 2:
                    return
                
                # The first byte after FF FF should be the target board ID, followed by the message
                hex_data = parts[1].strip().replace(" ", "")
                
                if len(hex_data) < 2:
                    return
                    
                # Get the target board ID
                target_id = int(hex_data[0:2], 16)
                
                # Check if message is for this board (target_id == 0 means broadcast)
                if target_id != 0 and target_id != self.board_id:
                    return  # Message is for a different board
                
                # Convert hex to ASCII (skipping the first byte which is the target ID)
                try:
                    ascii_message = bytes.fromhex(hex_data[2:]).decode('ascii', errors='replace')
                    
                    # Update last message received
                    self.last_pc_message = ascii_message
                    self.last_pc_seen = time.time()
                    
                    print("Received from PC (Target: " + str(target_id) + "): " + str(ascii_message))
                except Exception as e:
                    print("Error parsing data from PC: " + str(e))
                    
            except Exception as e:
                print("Error processing PC advertisement: " + str(e))
    
    def stop_scanning(self):
        self.running = False
        if self.scan_process:
            self.scan_process.terminate()
            self.scan_process = None
        if self.dump_process:
            self.dump_process.terminate()
            self.dump_process = None

# Simulated sensor function - replace with actual EV3 sensors
def read_sensor():
    # Here you would use the ev3dev Python API to read actual sensors
    # For example:
    # from ev3dev2.sensor import *
    # from ev3dev2.sensor.lego import TouchSensor, ColorSensor
    # touch = TouchSensor()
    # color = ColorSensor()
    # return touch.is_pressed, color.reflected_light_intensity
    
    # For simulation, return random values
    temp = 20 + random.random() * 10
    light = 30 + random.random() * 50
    return round(temp, 1), round(light, 1)

def main():
    print("Starting EV3 BLE Node " + str(BOARD_ID))
    
    # Check if running as root
    if os.geteuid() != 0:
        print("This script must be run as root. Try using sudo.")
        sys.exit(1)
    
    # Check for required tools
    if not shutil.which('hcitool') and not shutil.which('bluetoothctl'):
        print("Error: Neither hcitool nor bluetoothctl found!")
        print("Please install bluez tools: sudo apt-get install bluez")
        sys.exit(1)
    
    # Initialize BLE advertisement
    advertiser = BLEAdvertisement(BOARD_ID)
    if not advertiser.setup():
        print("Failed to set up BLE advertisement, continuing with limited functionality")
    
    # Initialize scanner
    scanner = BLEScanner(BOARD_ID)
    scanner.start_scanning()
    
    # Also start the simple scanner as a fallback
    simple_scanner = SimpleBLEDeviceScanner(BOARD_ID)
    simple_scanner.start_scanning()
    
    message_counter = 0
    try:
        while True:
            # Read sensor data
            temp, light = read_sensor()
            
            # Create message with sensor data
            message = "S" + str(BOARD_ID) + ":" + str(temp) + "," + str(light)
            print("Broadcasting: " + message)
            
            # Start advertising (or try to)
            advertiser.start_advertising(message)
            
            # Display last message from PC
            if scanner.last_pc_seen > 0:
                time_ago = time.time() - scanner.last_pc_seen
                print("Last PC message: " + scanner.last_pc_message + " (" + str(round(time_ago, 1)) + "s ago)")
            else:
                print("No PC messages received yet")
            
            # Wait between broadcasts
            time.sleep(5)
            message_counter += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        advertiser.stop_advertising()
        scanner.stop_scanning()
        simple_scanner.stop_scanning()
        print("EV3 BLE Node stopped")

if __name__ == "__main__":
    main()