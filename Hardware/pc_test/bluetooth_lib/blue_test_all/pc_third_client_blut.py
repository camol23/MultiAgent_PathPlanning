#!/usr/bin/env python3

import bluetooth
import threading
import time

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
    
    def start(self):
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
        while self.running:
            cycle_counter += 1
            print("\n--- Polling Cycle", cycle_counter, "---")
            
            # Loop through each EV3 device
            for device in EV3_DEVICES:
                device_id = device["id"]
                device_addr = device["address"]
                
                if not self.running:
                    break
                
                # Connect to this EV3
                success = self.connect_and_communicate(device_addr, device_id)
                
                if not success:
                    print("Failed to communicate with EV3", device_id)
                    # Wait a bit longer on failure before trying the next
                    time.sleep(0.5)
                
                # Small delay between devices (100-500ms as requested)
                time.sleep(0.2)
            
            # Wait before next polling cycle (keep cycle time in requested range)
            time.sleep(0.1)
    
    def connect_and_communicate(self, address, ev3_id):
        """Connect to a specific EV3, exchange data, then disconnect"""
        
        sock = None
        try:
            # Create socket
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Set connection timeout
            sock.settimeout(2.0)
            
            # Connect to EV3 server
            print("Connecting to EV3", ev3_id, "at", address)
            sock.connect((address, PORT))
            print("Connected to EV3", ev3_id)
            
            # Allow time for connection to stabilize
            time.sleep(0.1)
            
            # Initial handshake
            hello_msg = "HELLO_PC_" + str(ev3_id)
            sock.send(hello_msg.encode())
            print("Sent to EV3", ev3_id, ":", hello_msg)
            
            # Try to receive response
            try:
                data = sock.recv(1024)
                if data:
                    message = data.decode()
                    print("Received from EV3", ev3_id, ":", message)
            except bluetooth.BluetoothError as be:
                print("Bluetooth error during initial exchange:", be)
                return False
            except Exception as e:
                print("No initial response from EV3", ev3_id, ":", e)
                return False
            
            # Allow a short time between messages
            time.sleep(0.1)
            
            # Send a command
            with self.lock:
                command = "CMD_" + str(self.command_counter)
                self.command_counter += 1
            
            sock.send(command.encode())
            print("Sent command to EV3", ev3_id, ":", command)
            
            # Wait for acknowledgment
            try:
                sock.settimeout(1.0)
                data = sock.recv(1024)
                if data:
                    ack = data.decode()
                    print("Received from EV3", ev3_id, ":", ack)
            except bluetooth.BluetoothError as be:
                print("Bluetooth error waiting for ack:", be)
                return False
            except Exception as e:
                print("No command acknowledgment from EV3", ev3_id, ":", e)
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
                    # Allow some time before closing
                    time.sleep(0.1)
                    sock.shutdown(2)  # Shutdown in both directions
                    sock.close()
                    print("Closed connection to EV3", ev3_id)
                except:
                    pass
    
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