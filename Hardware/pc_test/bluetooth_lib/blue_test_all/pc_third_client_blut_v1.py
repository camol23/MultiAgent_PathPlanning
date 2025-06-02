#!/usr/bin/env python3

import bluetooth
import threading
import time
import subprocess
import os

# Define your EV3 devices - add all device addresses here
EV3_DEVICES = [
    {"address": "F0:45:DA:11:92:74", "id": 1},  # First EV3
    # {"address": "F0:45:DA:D2:C8:88", "id": 2},  # Second EV3
    # {"address": "00:16:53:ZZ:ZZ:ZZ", "id": 3}   # Third EV3
]

# RFCOMM port (must match port on EV3s)
PORT = 5

class PCController:
    def __init__(self):
        self.running = False
        self.connected_devices = {}
        self.command_counter = 0
        self.lock = threading.Lock()
        self.reset_counter = 0
    
    def reset_bluetooth(self):
        """Reset Bluetooth service to clear any stale connections"""
        try:
            self.reset_counter += 1
            print(f"Resetting Bluetooth service (Reset #{self.reset_counter})...")
            
            # Try to restart Bluetooth service - may require sudo
            try:
                subprocess.run(["sudo", "systemctl", "restart", "bluetooth.service"], timeout=5)
                time.sleep(2)  # Wait for service to restart
            except:
                print("Could not restart Bluetooth service (may need sudo), continuing anyway")
            
            # Kill any existing rfcomm processes
            try:
                subprocess.run(["pkill", "-f", "rfcomm"], timeout=5)
            except:
                pass
                
            time.sleep(1)  # Wait for processes to terminate
            print("Bluetooth reset complete")
            return True
        except Exception as e:
            print("Error resetting Bluetooth:", e)
            return False
    
    def start(self):
        # Reset Bluetooth before starting
        self.reset_bluetooth()
        
        self.running = True
        
        # Start polling thread to communicate with EV3s
        polling_thread = threading.Thread(target=self.poll_ev3_devices)
        polling_thread.daemon = True
        polling_thread.start()
        
        print("PC controller started. Will poll", len(EV3_DEVICES), "EV3 devices.")
        return True
    
    def poll_ev3_devices(self):
        """Continuously poll each EV3 device in sequence"""
        
        cycle_counter = 0
        consecutive_failures = 0
        
        while self.running:
            cycle_counter += 1
            print("\n--- Polling Cycle", cycle_counter, "---")
            
            cycle_success = False  # Flag to track if any device communication succeeds
            
            # Loop through each EV3 device
            for device in EV3_DEVICES:
                if not self.running:
                    break
                    
                device_id = device["id"]
                device_addr = device["address"]
                
                # Connect to this EV3
                success = self.connect_and_communicate(device_addr, device_id)
                
                if success:
                    cycle_success = True
                    consecutive_failures = 0  # Reset failure counter on success
                else:
                    print("Failed to communicate with EV3", device_id)
                
                # Small delay between devices
                time.sleep(0.3)
            
            # If no successful communications in this cycle, increment failure counter
            if not cycle_success:
                consecutive_failures += 1
                print(f"Complete polling cycle failed. Consecutive failures: {consecutive_failures}")
                
                # After 3 consecutive failed cycles, reset Bluetooth
                if consecutive_failures >= 3:
                    print("Too many consecutive failures. Resetting Bluetooth...")
                    self.reset_bluetooth()
                    consecutive_failures = 0  # Reset counter after reset
                    time.sleep(2)  # Wait for reset to complete
            
            # Wait before next polling cycle
            time.sleep(0.5)
    
    def connect_and_communicate(self, address, ev3_id):
        """Connect to a specific EV3, exchange data, then disconnect"""
        
        sock = None
        try:
            # Create socket
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Set connection timeout
            sock.settimeout(5.0)
            
            # Connect to EV3 server
            print("Connecting to EV3", ev3_id, "at", address)
            sock.connect((address, PORT))
            print("Connected to EV3", ev3_id)
            
            # Initial handshake - keep this simple
            hello_msg = "HELLO_PC_" + str(ev3_id)
            sock.send(hello_msg.encode())
            print("Sent to EV3", ev3_id, ":", hello_msg)
            
            # Try to receive response
            try:
                data = sock.recv(1024)
                if data:
                    message = data.decode()
                    print("Received from EV3", ev3_id, ":", message)
                else:
                    print("Empty response from EV3", ev3_id)
                    return False
            except Exception as e:
                print("No response from EV3", ev3_id, ":", e)
                return False
            
            # Send a command
            with self.lock:
                command = "CMD_" + str(self.command_counter)
                self.command_counter += 1
            
            sock.send(command.encode())
            print("Sent command to EV3", ev3_id, ":", command)
            
            # Wait for acknowledgment
            try:
                data = sock.recv(1024)
                if data:
                    ack = data.decode()
                    print("Received from EV3", ev3_id, ":", ack)
                else:
                    print("Empty acknowledgment from EV3", ev3_id)
                    return False
            except Exception as e:
                print("No acknowledgment from EV3", ev3_id, ":", e)
                return False
            
            return True
            
        except bluetooth.BluetoothError as be:
            print("Bluetooth error with EV3", ev3_id, ":", be)
            return False
        except Exception as e:
            print("Error communicating with EV3", ev3_id, ":", e)
            return False
        finally:
            # Always close socket properly
            if sock:
                try:
                    sock.close()
                    print("Closed connection to EV3", ev3_id)
                except Exception as e:
                    print("Error closing socket:", e)
    
    def stop(self):
        self.running = False
        time.sleep(1)  # Give threads time to terminate
        print("Controller stopped")

def main():
    controller = PCController()
    
    if not controller.start():
        print("Failed to start controller")
        return
    
    try:
        # Keep main thread running
        print("Controller running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()