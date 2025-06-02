#!/usr/bin/env python3
import subprocess
import time
import threading
import dbus
import dbus.service
import dbus.mainloop.glib
try:
    from gi.repository import GLib
except ImportError:
    import glib as GLib
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

# Check BlueZ version and capabilities
def check_bluez_version():
    try:
        output = subprocess.check_output(["bluetoothctl", "--version"], 
                                        stderr=subprocess.STDOUT).decode().strip()
        # print(f"BlueZ version: {output}")
        print("BlueZ version: ",output)
        return output
    except:
        print("Unable to determine BlueZ version")
        return "unknown"

class BluetoothManager:
    def __init__(self, board_id):
        self.board_id = board_id
        self.bus = None
        self.adapter = None
        self.running = False
        self.advertising_supported = False
        self.advertiser = None
        self.scanner = None
        self.last_pc_message = "No messages"
        self.last_pc_seen = 0
        
    def initialize(self):
        # Initialize D-Bus
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()
        self.mainloop = GLib.MainLoop()
        
        # Find the adapter
        self.adapter_path = self.find_adapter()
        if not self.adapter_path:
            print("Bluetooth adapter not found")
            return False
            
        # print(f"Found Bluetooth adapter at {self.adapter_path}")
        print("Found Bluetooth adapter at ", self.adapter_path)
        
        # Check for advertising support (safely)
        self.check_advertising_support()
        
        # Initialize components
        if self.advertising_supported:
            self.advertiser = SimpleAdvertiser(self.bus, self.adapter_path, self.board_id)
        
        # Always initialize scanner (works on most systems)
        self.scanner = MultipleScanner(self.board_id)
        
        return True
        
    def find_adapter(self):
        try:
            remote_om = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, "/"),
                                      "org.freedesktop.DBus.ObjectManager")
            objects = remote_om.GetManagedObjects()
            
            for obj_path, interfaces in objects.items():
                if ADAPTER_INTERFACE in interfaces:
                    return obj_path
        except Exception as e:
            # print(f"Error finding adapter: {e}")
            print("Error finding adapter: ",e)
        
        return None
    
    def check_advertising_support(self):
        try:
            adapter_obj = self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter_path)
            adapter_ifaces = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Introspectable")
            xml_data = adapter_ifaces.Introspect()
            
            # Check if LEAdvertisingManager1 is in the interfaces
            if "org.bluez.LEAdvertisingManager1" in xml_data:
                print("Bluetooth LE advertising is supported")
                self.advertising_supported = True
            else:
                print("Bluetooth LE advertising not supported on this device")
                self.advertising_supported = False
        except Exception as e:
            # print(f"Error checking advertising support: {e}")
            print("Error checking advertising support: ",e)
            self.advertising_supported = False
    
    def start(self):
        self.running = True
        
        # Start the mainloop in a separate thread
        self.mainloop_thread = threading.Thread(target=self._run_mainloop)
        self.mainloop_thread.daemon = True
        self.mainloop_thread.start()
        
        # Start the scanner
        self.scanner.start_scanning()
        
        print("Bluetooth manager started")
        
    def _run_mainloop(self):
        try:
            self.mainloop.run()
        except Exception as e:
            # print(f"Error in mainloop: {e}")
            print("Error in mainloop: ",e)
    
    def send_message(self, message):
        if self.advertising_supported and self.advertiser:
            return self.advertiser.send_message(message)
        else:
            # Fall back to alternative method
            # print(f"Would broadcast (no advertising support): {message}")
            print("Would broadcast (no advertising support): ",message)
            return self._send_message_alternative(message)
    
    def _send_message_alternative(self, message):
        # Alternative broadcasting method using system tools
        try:
            # device_name = f"EV3-{self.board_id}-{message[:10]}"  # Embed part of message in name
            device_name = "EV3-"+str(self.board_id)+"-"+str(message[:10]) 
            subprocess.run(["hciconfig", "hci0", "name", device_name], 
                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            # print(f"Error in alternative message sending: {e}")
            print("Error in alternative message sending: ",e)
            return False
    
    def get_latest_pc_message(self):
        if self.scanner:
            self.last_pc_message = self.scanner.last_pc_message
            self.last_pc_seen = self.scanner.last_pc_seen
        return self.last_pc_message, self.last_pc_seen
    
    def stop(self):
        self.running = False
        
        # Stop components
        if self.scanner:
            self.scanner.stop_scanning()
            
        if self.advertiser:
            self.advertiser.stop()
            
        # Stop mainloop
        if hasattr(self, 'mainloop') and self.mainloop.is_running():
            self.mainloop.quit()

class SimpleAdvertiser:
    def __init__(self, bus, adapter_path, board_id):
        self.bus = bus
        self.adapter_path = adapter_path
        self.board_id = board_id
        self.current_message = ""
        
        try:
            adapter_obj = self.bus.get_object(BLUEZ_SERVICE_NAME, adapter_path)
            self.ad_manager = dbus.Interface(adapter_obj, LE_ADVERTISING_MANAGER_INTERFACE)
        except Exception as e:
            # print(f"Error creating advertiser: {e}")
            print("Error creating advertiser: ",e)
            self.ad_manager = None
    
    def send_message(self, message):
        if not self.ad_manager:
            return False
            
        self.current_message = message
        
        # Use a simpler approach - change the device name to include the message
        # success = self._update_device_name(f"EV3{self.board_id}-{message[:15]}")
        success = self._update_device_name("EV3"+str(self.board_id)+"-"+str(message[:15]))
        
        # Try to broadcast data if possible
        try:
            # Simplified advertisement registration - less prone to errors
            self._broadcast_data(message)
        except Exception as e:
            # print(f"Broadcast error (non-critical): {e}")
            print("Broadcast error (non-critical): ",e)
            
        return success
    
    def _update_device_name(self, name):
        try:
            adapter_obj = self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter_path)
            props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            props.Set(ADAPTER_INTERFACE, "Alias", name)
            return True
        except Exception as e:
            # print(f"Error setting device name: {e}")
            print("Error setting device name: ",e)
            return False
    
    def _broadcast_data(self, message):
        # This is a fallback method that tries to use the advertising API safely
        try:
            # Create a simple advertisement object
            # path = f"/org/bluez/example/advertisement{self.board_id}"
            path = "/org/bluez/example/advertisement"+str(self.board_id)
            
            # Create advertisement data with manufacturer data
            ad_props = {
                "Type": dbus.String("broadcast"),
                "LocalName": dbus.String("EV3-"+str(self.board_id)), # "LocalName": dbus.String(f"EV3-{self.board_id}"),
                "ManufacturerData": dbus.Dictionary({
                    dbus.UInt16(0xFFFF): dbus.Array([
                        dbus.Byte(self.board_id),  # Board ID as first byte
                        *[dbus.Byte(ord(c)) for c in message]  # Message bytes
                    ], signature="y")
                }, signature="qv")
            }
            
            # Register advertisement, but don't fail if it doesn't work
            self.ad_manager.RegisterAdvertisement(
                path, 
                dbus.Dictionary(ad_props, signature="sv")
            )
        except Exception as e:
            # This is expected to fail on some systems
            pass
    
    def stop(self):
        # Reset device name
        try:
            self._update_device_name("EV3-"+str(self.board_id)) #self._update_device_name(f"EV3-{self.board_id}")
        except:
            pass

class MultipleScanner:
    def __init__(self, board_id):
        self.board_id = board_id
        self.running = False
        self.last_pc_message = "No messages"
        self.last_pc_seen = 0
        
        # Check for available tools
        self.hcitool_available = shutil.which('hcitool') is not None
        self.btmon_available = shutil.which('btmon') is not None
        self.hcidump_available = shutil.which('hcidump') is not None
        self.bluetoothctl_available = shutil.which('bluetoothctl') is not None
        
        print("Available tools:", 
             "hcitool" if self.hcitool_available else "",
             "btmon" if self.btmon_available else "",
             "hcidump" if self.hcidump_available else "",
             "bluetoothctl" if self.bluetoothctl_available else "")
    
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        
        # Start scan threads using available methods
        self.threads = []
        
        # Primary scanner
        if self.bluetoothctl_available:
            scanner_thread = threading.Thread(target=self._bluetoothctl_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            self.threads.append(scanner_thread)
        elif self.hcitool_available:
            scanner_thread = threading.Thread(target=self._hcitool_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            self.threads.append(scanner_thread)
        
        # Monitor threads if available
        if self.btmon_available:
            monitor_thread = threading.Thread(target=self._btmon_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            self.threads.append(monitor_thread)
        elif self.hcidump_available:
            monitor_thread = threading.Thread(target=self._hcidump_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            self.threads.append(monitor_thread)
        
        print("Started scanning with available methods")
    
    def _bluetoothctl_scan(self):
        try:
            while self.running:
                try:
                    proc = subprocess.Popen(
                        ["bluetoothctl"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                    )
                    
                    # Send scan command
                    proc.stdin.write(b"scan on\n")
                    proc.stdin.flush()
                    
                    # Process output
                    while self.running:
                        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        if "PC-Master" in line or "PC_Master" in line:
                            self._extract_message_from_name(line)
                except Exception as e:
                    # print(f"Bluetoothctl scan error: {e}")
                    print("Bluetoothctl scan error: ",e)
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("Bluetoothctl thread error: ",e)
    
    def _hcitool_scan(self):
        try:
            while self.running:
                try:
                    # Use lescan if available (better for advertisements)
                    proc = subprocess.Popen(
                        ["hcitool", "lescan", "--duplicates"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Process output
                    while self.running:
                        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        if "PC-Master" in line or "PC_Master" in line:
                            self._extract_message_from_name(line)
                except Exception as e:
                    # print(f"HCITool scan error: {e}")
                    print("HCITool scan error: ",e)
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("HCITool thread error: ", e)
    
    def _btmon_monitor(self):
        try:
            while self.running:
                try:
                    proc = subprocess.Popen(
                        ["btmon"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Process output
                    while self.running:
                        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        # Look for manufacturer data or device names
                        if "PC-Master" in line or "PC_Master" in line:
                            self._extract_message_from_monitor(line)
                            
                        # Look for manufacturer data
                        if "Manufacturer" in line and "ff ff" in line.lower():
                            self._extract_message_from_manufacturer_data(line)
                except Exception as e:
                    # print(f"BTMon monitor error: {e}")
                    print("BTMon monitor error: ",e)
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("BTMon thread error: ",e)
    
    def _hcidump_monitor(self):
        try:
            while self.running:
                try:
                    proc = subprocess.Popen(
                        ["hcidump", "--raw"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    buffer = b""
                    while self.running:
                        data = proc.stdout.read(1)
                        if not data:
                            break
                            
                        buffer += data
                        
                        if b"\n" in buffer:
                            lines = buffer.split(b"\n")
                            buffer = lines.pop()  # Keep incomplete line
                            
                            # Process complete lines
                            for line in lines:
                                line_str = line.decode('utf-8', errors='replace').strip()
                                if "ff ff" in line_str.lower():
                                    self._extract_message_from_raw_data(line_str)
                except Exception as e:
                    # print(f"HCIDump monitor error: {e}")
                    print("HCIDump monitor error: ", e)
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            # print(f"HCIDump thread error: {e}")
            print("HCIDump thread error: ",e)
    
    def _extract_message_from_name(self, line):
        try:
            # Format could be like "Device XX:XX:XX:XX:XX:XX PC-Master-BoardX-Message"
            if "PC-Master" in line:
                parts = line.split("PC-Master-")
                if len(parts) < 2:
                    return
                
                data_part = parts[1].strip()
                
                # Check if there's a board identifier
                if data_part.startswith("Board"):
                    # Format: BoardX-Message
                    board_parts = data_part.split("-", 1)
                    if len(board_parts) < 2:
                        return
                        
                    target_board = board_parts[0][5:]  # Extract number after "Board"
                    message = board_parts[1]
                    
                    # Check if message is for this board
                    try:
                        target_id = int(target_board)
                        if target_id != 0 and target_id != self.board_id:
                            return  # Not for this board
                    except:
                        # If we can't parse the board ID, assume broadcast
                        pass
                    
                    self._update_message(message)
                else:
                    # Direct message without board ID
                    self._update_message(data_part)
        except Exception as e:
            # print(f"Error extracting message from name: {e}")
            print("Error extracting message from name: ",e)
    
    def _extract_message_from_monitor(self, line):
        # Extract message from btmon output
        try:
            if "Name:" in line and "PC-Master" in line:
                parts = line.split("PC-Master-")
                if len(parts) < 2:
                    return
                
                data_part = parts[1].strip()
                self._extract_message_from_name("PC-Master-" + data_part)
        except Exception as e:
            # print(f"Error extracting from monitor: {e}")
            print("Error extracting from monitor: ", e)
    
    def _extract_message_from_manufacturer_data(self, line):
        try:
            # Format: "Manufacturer: FFFF" followed by data
            if "ff ff" not in line.lower():
                return
                
            parts = line.lower().split("ff ff")
            if len(parts) < 2:
                return
            
            # Extract bytes after FFFF
            data_part = parts[1].strip()
            
            # Try to extract board ID and message
            hex_values = data_part.split()
            if not hex_values:
                return
            
            # First byte should be board ID
            try:
                target_id = int(hex_values[0], 16)
                
                # Check if for this board
                if target_id != 0 and target_id != self.board_id:
                    return  # Not for us
                
                # Convert remaining bytes to message
                message_bytes = bytes([int(h, 16) for h in hex_values[1:]])
                message = message_bytes.decode('ascii', errors='replace')
                
                self._update_message(message)
            except Exception as e:
                # print(f"Error parsing manufacturer data: {e}")
                print("Error parsing manufacturer data: ",e)
        except Exception as e:
            # print(f"Error in manufacturer data extraction: {e}")
            print("Error in manufacturer data extraction: ",e)
    
    def _extract_message_from_raw_data(self, raw_data):
        try:
            # Check for our manufacturer ID (0xFFFF)
            if "ff ff" not in raw_data.lower():
                return
                
            parts = raw_data.lower().split("ff ff")
            if len(parts) < 2:
                return
            
            # Try to parse hexadecimal data
            hex_data = parts[1].replace(" ", "").strip()
            
            if len(hex_data) < 2:
                return
                
            # First byte is board ID
            target_id = int(hex_data[0:2], 16)
            
            # Check if for this board
            if target_id != 0 and target_id != self.board_id:
                return  # Not for us
            
            # Convert remaining hex to ASCII
            try:
                message_bytes = bytes.fromhex(hex_data[2:])
                message = message_bytes.decode('ascii', errors='replace')
                
                self._update_message(message)
            except Exception as e:
                # print(f"Error decoding raw data: {e}")
                print("Error decoding raw data: ", e)
        except Exception as e:
            print("Error processing raw data: ", e)
    
    def _update_message(self, message):
        # print(f"Received from PC: {message}")
        print("Received from PC: "+str(message))
        self.last_pc_message = message
        self.last_pc_seen = time.time()
    
    def stop_scanning(self):
        self.running = False

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
    
    # Check BlueZ version
    check_bluez_version()
    
    # Initialize Bluetooth manager
    bt_manager = BluetoothManager(BOARD_ID)
    if not bt_manager.initialize():
        print("Failed to initialize Bluetooth manager, exiting")
        sys.exit(1)
    
    # Start the manager
    bt_manager.start()
    
    message_counter = 0
    try:
        while True:
            # Read sensor data
            temp, light = read_sensor()
            
            # Create message with sensor data
            message = "S" + str(BOARD_ID) + ":" + str(temp) + "," + str(light)
            print("Broadcasting: " + message)
            
            # Send message
            bt_manager.send_message(message)
            
            # Display last message from PC
            pc_message, pc_time = bt_manager.get_latest_pc_message()
            if pc_time > 0:
                time_ago = time.time() - pc_time
                print("Last PC message: " + pc_message + " (" + str(round(time_ago, 1)) + "s ago)")
            else:
                print("No PC messages received yet")
            
            # Wait between broadcasts
            time.sleep(5)
            message_counter += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        bt_manager.stop()
        print("EV3 BLE Node stopped")

if __name__ == "__main__":
    main()