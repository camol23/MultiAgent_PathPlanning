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

# BlueZ service names
BLUEZ_SERVICE_NAME = "org.bluez"
ADAPTER_INTERFACE = "org.bluez.Adapter1"

class BluetoothManager:
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
        
        tools_available = []
        if self.hcitool_available:
            tools_available.append("hcitool")
        if self.btmon_available:
            tools_available.append("btmon")
        if self.hcidump_available:
            tools_available.append("hcidump")
        if self.bluetoothctl_available:
            tools_available.append("bluetoothctl")
            
        print("Available tools:", " ".join(tools_available))
    
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
            
        print("Found Bluetooth adapter at " + self.adapter_path)
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
            print("Error finding adapter: " + str(e))
        
        return None
    
    def start(self):
        self.running = True
        
        # Start the mainloop in a thread
        self.mainloop_thread = threading.Thread(target=self._run_mainloop)
        self.mainloop_thread.daemon = True
        self.mainloop_thread.start()
        
        # Start scanner and broadcaster threads
        self.start_scanning()
        
        print("Bluetooth manager started")
        
    def _run_mainloop(self):
        try:
            self.mainloop.run()
        except Exception as e:
            print("Error in mainloop: " + str(e))
    
    def start_scanning(self):
        # Start scanner threads using available methods
        self.scanner_threads = []
        
        # Primary scanner
        if self.bluetoothctl_available:
            scanner_thread = threading.Thread(target=self._bluetoothctl_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            self.scanner_threads.append(scanner_thread)
        elif self.hcitool_available:
            scanner_thread = threading.Thread(target=self._hcitool_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            self.scanner_threads.append(scanner_thread)
        
        # Monitor threads if available
        if self.btmon_available:
            monitor_thread = threading.Thread(target=self._btmon_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            self.scanner_threads.append(monitor_thread)
        elif self.hcidump_available:
            monitor_thread = threading.Thread(target=self._hcidump_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            self.scanner_threads.append(monitor_thread)
    
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
                    print("Bluetoothctl scan error: " + str(e))
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("Bluetoothctl thread error: " + str(e))
    
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
                    print("HCITool scan error: " + str(e))
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("HCITool thread error: " + str(e))
    
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
                            
                        # Look for device names or manufacturer data
                        if "PC-Master" in line or "PC_Master" in line:
                            self._extract_message_from_monitor(line)
                except Exception as e:
                    print("BTMon monitor error: " + str(e))
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("BTMon thread error: " + str(e))
    
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
                                if "PC-Master" in line_str:
                                    self._extract_message_from_raw_data(line_str)
                except Exception as e:
                    print("HCIDump monitor error: " + str(e))
                    time.sleep(5)  # Wait before retry
                
                time.sleep(1)
        except Exception as e:
            print("HCIDump thread error: " + str(e))
    
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
            print("Error extracting message from name: " + str(e))
    
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
            print("Error extracting from monitor: " + str(e))
    
    def _extract_message_from_raw_data(self, raw_data):
        try:
            # Look for PC-Master in the raw data
            if "PC-Master" not in raw_data:
                return
                
            self._extract_message_from_name(raw_data)
        except Exception as e:
            print("Error processing raw data: " + str(e))
    
    def _update_message(self, message):
        print("Received from PC: " + message)
        self.last_pc_message = message
        self.last_pc_seen = time.time()
    
    def send_message(self, message):
        """Send a message by changing the device name"""
        # Create the device name with embedded data
        device_name = "EV3-" + str(self.board_id) + "-" + message
        
        # Truncate if necessary to stay within Bluetooth name limits
        if len(device_name) > 30:
            device_name = device_name[:30]
        
        success = False
        
        # Try to set the device name using D-Bus
        try:
            adapter_obj = self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter_path)
            props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            props.Set(ADAPTER_INTERFACE, "Alias", device_name)
            success = True
        except Exception as e:
            print("Error setting device name via D-Bus: " + str(e))
            
            # Fall back to using hciconfig
            try:
                cmd = ["hciconfig", "hci0", "name", device_name]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                success = result.returncode == 0
            except Exception as e2:
                print("Error setting device name via hciconfig: " + str(e2))
                
                # Last resort: try bluetoothctl
                if self.bluetoothctl_available:
                    try:
                        proc = subprocess.Popen(
                            ["bluetoothctl"], 
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT
                        )
                        
                        cmd = "system-alias " + device_name + "\n"
                        proc.stdin.write(cmd.encode())
                        proc.stdin.flush()
                        time.sleep(0.5)
                        
                        proc.stdin.write(b"quit\n")
                        proc.stdin.flush()
                        
                        success = True
                    except Exception as e3:
                        print("Error setting name via bluetoothctl: " + str(e3))
        
        print("Broadcasting as: " + device_name + " (Success: " + str(success) + ")")
        return success
    
    def stop(self):
        self.running = False
        
        # Reset device name
        try:
            adapter_obj = self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter_path)
            props = dbus.Interface(adapter_obj, "org.freedesktop.DBus.Properties")
            props.Set(ADAPTER_INTERFACE, "Alias", "EV3-" + str(self.board_id))
        except:
            pass
            
        # Stop mainloop
        if hasattr(self, 'mainloop') and self.mainloop.is_running():
            self.mainloop.quit()

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
    # temp = 20 + random.random() * 10
    # light = 30 + random.random() * 50
    temp = 10
    light = 8
    return round(temp, 1), round(light, 1)

def main():
    print("Starting EV3 BLE Node " + str(BOARD_ID))
    
    # Check if running as root
    if os.geteuid() != 0:
        print("This script must be run as root. Try using sudo.")
        sys.exit(1)
    
    # Check BlueZ version
    try:
        output = subprocess.check_output(["bluetoothctl", "--version"], 
                                        stderr=subprocess.STDOUT).decode().strip()
        print("BlueZ version: " + output)
    except:
        print("Unable to determine BlueZ version")
    
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
            if bt_manager.last_pc_seen > 0:
                time_ago = time.time() - bt_manager.last_pc_seen
                print("Last PC message: " + bt_manager.last_pc_message + " (" + str(round(time_ago, 1)) + "s ago)")
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