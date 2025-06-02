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
        self.hciconfig_available = shutil.which('hciconfig') is not None
        
        tools = []
        if self.bluetoothctl_available:
            tools.append("bluetoothctl")
        if self.hcitool_available:
            tools.append("hcitool")
        if self.hciconfig_available:
            tools.append("hciconfig")
            
        print("Available tools: " + " ".join(tools))
        
        if not self.bluetoothctl_available and not self.hcitool_available and not self.hciconfig_available:
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
        # Start multiple scanning methods in parallel for redundancy
        threads = []
        
        if self.bluetoothctl_available:
            scanner_thread = threading.Thread(target=self._bluetoothctl_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            threads.append(scanner_thread)
            
        if self.hcitool_available:
            scanner_thread = threading.Thread(target=self._hcitool_scan)
            scanner_thread.daemon = True
            scanner_thread.start()
            threads.append(scanner_thread)
            
        # Also periodically check device names using hciconfig
        if self.hciconfig_available:
            name_checker = threading.Thread(target=self._check_device_names)
            name_checker.daemon = True
            name_checker.start()
            threads.append(name_checker)
            
        # Wait for threads
        for thread in threads:
            thread.join()
    
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
                # Try both regular scan and lescan alternately
                
                # Regular scan
                try:
                    output = subprocess.check_output(
                        ["hcitool", "scan"],
                        stderr=subprocess.STDOUT
                    ).decode('utf-8', errors='replace')
                    
                    for line in output.splitlines():
                        if "EV3-" in line:
                            self._process_ev3_device(line)
                except Exception as e:
                    print("HCITool scan error: " + str(e))
                
                # Try LE scan if device didn't exit immediately
                try:
                    proc = subprocess.Popen(
                        ["hcitool", "lescan"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Read for up to 10 seconds
                    end_time = time.time() + 10
                    while time.time() < end_time and self.running:
                        line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                            
                        if "EV3-" in line:
                            self._process_ev3_device(line)
                            
                    # Kill the process after our timeout
                    proc.terminate()
                except Exception as e:
                    # LE scan might not be supported - that's OK
                    pass
                    
                # Get names of already paired devices
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
                    
                time.sleep(5)
        except Exception as e:
            print("HCITool thread error: " + str(e))
    
    def _check_device_names(self):
        """Periodically check device names using native commands"""
        try:
            while self.running:
                try:
                    # Use bluetoothctl to get device list
                    if self.bluetoothctl_available:
                        proc = subprocess.Popen(
                            ["bluetoothctl"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT
                        )
                        
                        # Get devices
                        proc.stdin.write(b"devices\n")
                        proc.stdin.flush()
                        time.sleep(0.5)
                        
                        # Get output
                        proc.stdin.write(b"quit\n")
                        proc.stdin.flush()
                        output = proc.stdout.read().decode('utf-8', errors='replace')
                        
                        # Process each line
                        for line in output.splitlines():
                            if "Device" in line and "EV3-" in line:
                                self._process_ev3_device(line)
                except Exception as e:
                    print("Error checking device names: " + str(e))
                    
                time.sleep(5)  # Check every 5 seconds
        except Exception as e:
            print("Device name checker thread error: " + str(e))
    
    def _process_ev3_device(self, line):
        try:
            # First, check if line contains "EV3-"
            if "EV3-" not in line:
                return
                
            # Extract device name
            ev3_part = None
            parts = line.split()
            
            # Search for the EV3 part
            for part in parts:
                if part.startswith("EV3-"):
                    ev3_part = part
                    break
                    
            if not ev3_part:
                return
                
            # Extract board ID
            try:
                # Format should be EV3-1-S1:25.2,45.6
                # Or EV3-1 (just the ID)
                board_id_str = ev3_part[4:5]  # Extract the number after "EV3-"
                board_id = int(board_id_str)
                
                # Check if board_id is valid
                if board_id < 1 or board_id > 3:
                    print("Invalid board ID: " + str(board_id))
                    return
                    
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
                                # Invalid number format
                                pass
            except (ValueError, IndexError) as e:
                # If we can't extract the board ID or it's invalid, just log
                print("Error extracting board ID: " + str(e) + " from " + ev3_part)
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
        
        # Try all available methods to set device name
        device_name = PC_NAME + "-Board" + str(board_id) + "-" + message
        
        # Method 1: hciconfig (most reliable)
        if self.hciconfig_available:
            try:
                result = subprocess.run(
                    ["hciconfig", "hci0", "name", device_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                success = result.returncode == 0
                if success:
                    print("Set name using hciconfig: " + device_name)
                    return True
            except Exception as e:
                print("Error setting device name via hciconfig: " + str(e))
        
        # Method 2:
        # Method 2: bluetoothctl
        if not success and self.bluetoothctl_available:
            try:
                proc = subprocess.Popen(
                    ["bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                
                # Try both commands for setting name
                cmd = "system-alias " + device_name + "\n"
                proc.stdin.write(cmd.encode())
                proc.stdin.flush()
                time.sleep(0.5)
                
                # Also try set-alias as fallback
                cmd = "set-alias " + device_name + "\n"
                proc.stdin.write(cmd.encode())
                proc.stdin.flush()
                time.sleep(0.5)
                
                # Read response
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
                output = proc.stdout.read().decode('utf-8', errors='replace')
                
                success = "successful" in output.lower() or "done" in output.lower() or "succeeded" in output.lower()
                if success:
                    print("Set name using bluetoothctl: " + device_name)
                    return True
            except Exception as e:
                print("Error setting device name via bluetoothctl: " + str(e))
        
        # Method 3: hostnamectl (may require sudo)
        if not success:
            try:
                cmd = "sudo hostnamectl set-hostname " + device_name
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                success = result.returncode == 0
                if success:
                    print("Set name using hostnamectl: " + device_name)
                    return True
            except Exception as e:
                print("Error setting hostname: " + str(e))
        
        print("Message to board " + str(board_id) + ": " + message + " (Success: " + str(success) + ")")
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
            if self.hciconfig_available:
                subprocess.run(["hciconfig", "hci0", "name", PC_NAME])
            elif self.bluetoothctl_available:
                proc = subprocess.Popen(
                    ["bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                
                proc.stdin.write(("system-alias " + PC_NAME + "\n").encode())
                proc.stdin.flush()
                time.sleep(0.1)
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
        except Exception as e:
            print("Error resetting device name: " + str(e))

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
                try:
                    board_id = int(parts[1])
                    if board_id < 1 or board_id > 3:
                        print("Invalid board ID. Use 1, 2, or 3.")
                        continue
                    message = " ".join(parts[2:])
                    manager.send_message(board_id, message)
                except ValueError:
                    print("Invalid board ID. Must be a number.")
            elif parts[0] == "broadcast" and len(parts) >= 2:
                message = " ".join(parts[1:])
                manager.send_message(0, message)  # 0 = broadcast to all
            else:
                print("Unknown command. Use 'send <board_id> <message>' or 'broadcast <message>'.")
        except Exception as e:
            print("Error processing command: " + str(e))

def main():
    parser = argparse.ArgumentParser(description="PC Master for EV3 Bluetooth Communication")
    args = parser.parse_args()
    
    print("Starting PC-Master")
    
    # Check if running as root on Linux
    if os.name != 'nt' and os.geteuid() != 0:
        print("Warning: On Linux, this script should be run as root for full functionality.")
        print("Try running with: sudo python3 " + sys.argv[0])
    
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