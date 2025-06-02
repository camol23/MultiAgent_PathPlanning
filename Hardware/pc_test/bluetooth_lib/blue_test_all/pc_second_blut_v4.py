#!/usr/bin/env python3

import bluetooth
import threading
import time

# EV3 device MAC addresses (replace with your actual EV3 MAC addresses)
ev3_addresses = [
   "F0:45:DA:11:92:74",  # EV3 #
   "F0:45:DA:D2:C8:88"
]

# RFCOMM port (must match on EV3s)
port = 1

class EV3Connection:
    def __init__(self, mac_address, ev3_id):
        self.mac_address = mac_address
        self.ev3_id = ev3_id
        self.socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None
        self.socket_lock = threading.Lock()

    def connect(self):
        try:
            # Create new socket
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Set socket options before connecting
            self.socket.setblocking(True)
            
            # Connect with timeout
            print(f"Attempting to connect to EV3 #{self.ev3_id} at {self.mac_address}")
            self.socket.connect((self.mac_address, port))
            
            # Successful connection
            self.connected = True
            print(f"Connected to EV3 #{self.ev3_id}")
            
            # Allow connection to stabilize
            time.sleep(1.0)
            
            # Start receive thread
            self.stop_thread = False
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to connect to EV3 #{self.ev3_id}: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False

    def send_data(self, data):
        if not self.connected or not self.socket:
            return False

        with self.socket_lock:
            try:
                # Convert string to bytes if needed
                if isinstance(data, str):
                    data = data.encode()
                
                # Send with timeout
                self.socket.send(data)
                # Small delay after sending
                time.sleep(0.05)
                return True
            except Exception as e:
                print(f"Error sending to EV3 #{self.ev3_id}: {e}")
                self.connected = False
                return False

    def receive_data(self):
        buffer_size = 1024
        
        while not self.stop_thread and self.connected:
            with self.socket_lock:
                if not self.socket:
                    self.connected = False
                    break
                    
                try:
                    # Set a short timeout
                    self.socket.settimeout(0.5)
                    
                    # Try to receive data
                    data = self.socket.recv(buffer_size)
                    
                    # Reset timeout
                    self.socket.settimeout(None)
                    
                    # Process received data
                    if data:
                        try:
                            message = data.decode('utf-8')
                            print(f"Received from EV3 #{self.ev3_id}: {message}")
                        except UnicodeDecodeError:
                            print(f"Received binary data from EV3 #{self.ev3_id}")
                    else:
                        # Empty data means connection closed
                        print(f"Connection to EV3 #{self.ev3_id} closed by remote device")
                        self.connected = False
                        break
                        
                except bluetooth.btcommon.BluetoothError as e:
                    if str(e) == "timed out":
                        # Just a timeout, continue
                        pass
                    else:
                        print(f"Bluetooth error from EV3 #{self.ev3_id}: {e}")
                        self.connected = False
                        break
                except Exception as e:
                    print(f"Error receiving from EV3 #{self.ev3_id}: {e}")
                    self.connected = False
                    break
            
            # Sleep outside the lock
            time.sleep(0.1)
    
    def disconnect(self):
        self.stop_thread = True
        
        # Wait for thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(1.0)
        
        # Close socket
        with self.socket_lock:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
        
        self.connected = False
        print(f"Disconnected from EV3 #{self.ev3_id}")

def main():
    # Create connections to all EV3s
    ev3_connections = []
    
    # Try to connect to each EV3
    for i, addr in enumerate(ev3_addresses):
        conn = EV3Connection(addr, i+1)
        if conn.connect():
            # Wait for the initial message
            time.sleep(2.0)
            ev3_connections.append(conn)
    
    # Main communication loop
    try:
        send_counter = 0
        
        while ev3_connections:
            # Copy the list to avoid modification during iteration
            current_connections = ev3_connections.copy()
            still_connected = []
            
            for conn in current_connections:
                if conn.connected:
                    # Format message with counter to ensure uniqueness
                    message = f"CMD_{conn.ev3_id}_{send_counter}"
                    
                    if conn.send_data(message):
                        still_connected.append(conn)
                    else:
                        print(f"Lost connection to EV3 #{conn.ev3_id}")
                        conn.disconnect()
            
            # Update the connections list
            ev3_connections = still_connected
            
            # Increment counter
            send_counter += 1
            
            # If all connections are lost, exit loop
            if not ev3_connections:
                print("All connections lost. Exiting.")
                break
            
            # Wait before next send
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Clean up all connections
        for conn in ev3_connections:
            conn.disconnect()

if __name__ == "__main__":
    main()