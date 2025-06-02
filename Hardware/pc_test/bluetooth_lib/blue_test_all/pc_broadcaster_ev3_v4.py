#!/usr/bin/env python3
import subprocess
import time
import threading
import sys
import os
import shutil
import argparse

# Define constants
PC_NAME = "PC-Master"

class BluetoothManager:
    def __init__(self):
        self.running = False
        self.ev3_data = {
            1: {"last_seen": 0, "temp": 0, "light": 0, "status": "Disconnected"},
            2: {"last_seen": 0, "temp": 0, "light": 0, "status": "Disconnected"},
            3: {"last_seen": 0, "temp": 0, "light": 0, "status": "Disconnected"}
        }
        
        # Check available tools
        self.bluetoothctl_available = shutil.which('bluetoothctl') is not None
        self.hcitool_available = shutil.which('hcitool') is not None
        
        tools = []
        if self.bluetoothctl_available:
            tools.append("bluetoothctl")
        if self.hcitool_available:
            tools.append("hcitool")
            
        print("Available tools:", " ".join(tools))
        
        if not self.bluetoothctl_available and not self.hcitool_available:
            print("Error: No Bluetooth tools found. Please install bluez.")
            sys.exit(1)
    
    def start(self):
        # Start scanning for EV3 devices
        self.running = True
        
        # Start scanner thread
        self.scanner_thread = threading.Thread(target=self._scan_loop)
        self.scanner_thread.daemon = True
        self.scanner_thread.start()
        
        print("Bluetooth manager started")
    
    def _scan_loop(self):
        if self.bluetoothctl_available:
            scanner_thread = threading.Thread(target=self._bluetoothctl_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
        elif self.hcitool_available:
            scanner_thread = threading.Thread(target=self._hcitool_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
    
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
                            
                        # Look for EV3 devices
                        if "EV3-" in line:
                            self._process_ev3_device(line)
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
                    # Use lescan or regular scan
                    proc = subprocess.Popen(
                        ["hcitool", "scan"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    while self.running:
                        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        if "EV3-" in line:
                            self._process_ev3_device(line)
                            
                    # Wait a bit before rescanning
                    time.sleep(5)
                except Exception as e:
                    print("HCITool scan error: " + str(e))
                    time.sleep(5)  # Wait before retry
                
                # Also try to get names of already paired devices
                try:
                    output = subprocess.check_output(
                        ["hcitool", "dev"],
                        stderr=subprocess.STDOUT
                    ).decode('utf-8', errors='replace')
                    
                    for line in output.splitlines():
                        if line.startswith("\t"):
                            parts = line.strip().split("\t")
                            if len(parts) >= 2:
                                addr = parts[1]
                                name_output = subprocess.check_output(
                                    ["hcitool", "name", addr],
                                    stderr=subprocess.STDOUT
                                ).decode('utf-8', errors='replace').strip()
                                
                                if name_output and "EV3-" in name_output:
                                    self._process_ev3_device("Device " + addr + " " + name_output)
                except Exception as e:
                    print("Error getting device names: " + str(e))
        except Exception as e:
            print("HCITool thread error: " + str(e))
    
    def _process_ev3_device(self, line):
        try:
            # Extract device address and name
            parts = line.split()
            if len(parts) < 3 or "EV3-" not in line:
                return
                
            # Find the part starting with "EV3-"
            ev3_part = None
            for part in parts:
                if part.startswith("EV3-"):
                    ev3_part = part
                    break
                    
            if not ev3_part:
                return
                
            # Extract board ID and data
            try:
                # Format should be EV3-1-S1:25.2,45.6
                # Or EV3-1 (just the ID)
                board_id = int(ev3_part[4:5])  # Extract the number after "EV3-"
                
                # Update the last seen time
                self.ev3_data[board_id]["last_seen"] = time.time()
                self.ev3_data[board_id]["status"] = "Connected"
                
                # Check if there's sensor data
                if "-" in ev3_part and len(ev3_part) > 6:
                    data_part = ev3_part.split("-", 2)[2]  # Get part after EV3-1-
                    
                    if data_part.startswith("S") and ":" in data_part:
                        value_part = data_part.split(":", 1)[1]
                        if "," in value_part:
                            temp_str, light_str = value_part.split(",", 1)
                            try:
                                self.ev3_data[board_id]["temp"] = float(temp_str)
                                self.ev3_data[board_id]["light"] = float(light_str)
                                print("Received from EV3-" + str(board_id) + ": temp=" + 
                                      str(self.ev3_data[board_id]["temp"]) + ", light=" + 
                                      str(self.ev3_data[board_id]["light"]))
                            except ValueError:
                                pass  # Invalid number format
            except (ValueError, IndexError, KeyError) as e:
                # If we can't extract the board ID or it's invalid, just ignore
                print("Error processing device data: " + str(e))
        except Exception as e:
            print("Error processing EV3 device: " + str(e))
    
    def send_message(self, board_id, message):
        """Send a message to a specific EV3 board or all boards (board_id=0)"""
        if not message:
            return False
            
        # Limit message length to ensure it fits
        if len(message) > 15:
            message = message[:15]
            
        success = False
        
        # Try primary method: set device name with embedded message
        device_name = PC_NAME + "-Board" + str(board_id) + "-" + message
        
        try:
            # Set local device name to include the message
            if self.bluetoothctl_available:
                proc = subprocess.Popen(
                    ["bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                
                # Send command to set name
                cmd = "system-alias " + device_name + "\n"
                proc.stdin.write(cmd.encode())
                proc.stdin.flush()
                
                # Read response
                time.sleep(0.1)
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
                output = proc.stdout.read().decode('utf-8', errors='replace')
                
                success = "successful" in output.lower() or "done" in output.lower()
            elif self.hcitool_available:
                result = subprocess.run(
                    ["hciconfig", "hci0", "name", device_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                success = result.returncode == 0
        except Exception as e:
            print("Error setting device name: " + str(e))
            success = False
        
        print("Sent message to board " + str(board_id) + ": " + message + " (Success: " + str(success) + ")")
        return success
    
    def get_ev3_status(self):
        """Get the status of all EV3 boards"""
        current_time = time.time()
        
        # Update connection status based on when we last saw each board
        for board_id in self.ev3_data:
            last_seen = self.ev3_data[board_id]["last_seen"]
            if last_seen == 0:
                self.ev3_data[board_id]["status"] = "Never seen"
            elif current_time - last_seen > 15:  # If not seen in 15 seconds
                self.ev3_data[board_id]["status"] = "Disconnected"
        
        return self.ev3_data
    
    def stop(self):
        self.running = False
        
        # Reset device name
        try:
            if self.bluetoothctl_available:
                proc = subprocess.Popen(
                    ["bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                
                proc.stdin.write(b"system-alias PC-Master\n")
                proc.stdin.flush()
                time.sleep(0.1)
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
            elif self.hcitool_available:
                subprocess.run(["hciconfig", "hci0", "name", "PC-Master"])
        except:
            pass

def display_status(manager):
    """Display the status of all EV3 boards in a simple UI"""
    while manager.running:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("===============================================")
        print("             PC-Master Control                 ")
        print("===============================================")
        
        status = manager.get_ev3_status()
        for board_id in sorted(status.keys()):
            board = status[board_id]
            last_seen_str = "Never" if board["last_seen"] == 0 else str(int(time.time() - board["last_seen"])) + "s ago"
            
            print("EV3 Board " + str(board_id) + ": " + board["status"])
            print("  Last seen: " + last_seen_str)
            if board["status"] == "Connected":
                print("  Temperature: " + str(board["temp"]))
                print("  Light: " + str(board["light"]))
            print("-----------------------------------------------")
        
        print("Commands:")
        print("  send <board_id> <message> - Send message to a board")
        print("  broadcast <message> - Send to all boards")
        print("  quit - Exit the program")
        
        time.sleep(1)

def command_loop(manager):
    """Process user commands"""
    while manager.running:
        try:
            cmd = input("> ")
            parts = cmd.split()
            
            if not parts:
                continue
                
            if parts[0] == "quit" or parts[0] == "exit":
                manager.running = False
                break
            elif parts[0] == "send" and len(parts) >= 3:
                board_id = int(parts[1])
                message = " ".join(parts[2:])
                manager.send_message(board_id, message)
            elif parts[0] == "broadcast" and len(parts) >= 2:
                message = " ".join(parts[1:])
                manager.send_message(0, message)  # 0 = broadcast to all
            else:
                print("Unknown command")
        except Exception as e:
            print("Error processing command: " + str(e))

def main():
    parser = argparse.ArgumentParser(description="PC Master for EV3 Bluetooth Communication")
    args = parser.parse_args()
    
    print("Starting PC-Master")
    
    # Check if running as root on Linux
    if os.name != 'nt' and os.geteuid() != 0:
        print("Warning: On Linux, this script should be run as root for full functionality.")
    
    # Initialize Bluetooth manager
    manager = BluetoothManager()
    manager.start()
    
    # Start status display in a separate thread
    status_thread = threading.Thread(target=display_status, args=(manager,))
    status_thread.daemon = True
    status_thread.start()
    
    # Process user commands
    try:
        command_loop(manager)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        manager.stop()
        print("PC-Master stopped")

if __name__ == "__main__":
    main()