#!/usr/bin/env python3

import bluetooth
import threading
import time

# EV3 ID - set this differently for each EV3 (1, 2, or 3)
EV3_ID = 1

# PC Bluetooth address - IMPORTANT: Change this to your PC's Bluetooth MAC address
PC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address

# RFCOMM port (must match port on PC server)
PORT = 5

class EV3Client:
    def __init__(self):
        self.running = False
        self.connected = False
        self.socket = None
        self.data_counter = 0
        self.connect_retry_count = 0
    
    def start(self):
        self.running = True
        
        # Start connection thread
        connection_thread = threading.Thread(target=self.connection_manager)
        connection_thread.daemon = True
        connection_thread.start()
        
        print("EV3", EV3_ID, "client started. Will connect to PC at", PC_ADDRESS)
        return True
    
    def connection_manager(self):
        """Manages connection to the PC server"""
        
        while self.running:
            if not self.connected:
                success = self.connect_to_pc()
                
                if not success:
                    self.connect_retry_count += 1
                    retry_delay = min(self.connect_retry_count * 1.5, 10)
                    print("Connection failed. Retrying in", retry_delay, "seconds...")
                    time.sleep(retry_delay)
                else:
                    self.connect_retry_count = 0
            
            time.sleep(1)
    
    def connect_to_pc(self):
        """Establish connection to PC server"""
        
        sock = None
        try:
            # Create socket
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Set connection timeout
            sock.settimeout(5.0)
            
            # Connect to PC server
            print("Connecting to PC at", PC_ADDRESS)
            sock.connect((PC_ADDRESS, PORT))
            print("Connected to PC")
            
            # Store socket
            self.socket = sock
            self.connected = True
            
            # Send identification
            hello_msg = "HELLO_EV3_" + str(EV3_ID)
            sock.send(hello_msg.encode())
            print("Sent:", hello_msg)
            
            # Wait for response
            try:
                data = sock.recv(1024)
                if data:
                    response = data.decode()
                    print("Received:", response)
            except Exception as e:
                print("No initial response from PC:", e)
            
            # Start communication loop
            comm_thread = threading.Thread(target=self.communication_loop)
            comm_thread.daemon = True
            comm_thread.start()
            
            return True
            
        except Exception as e:
            print("Error connecting to PC:", e)
            if sock:
                try:
                    sock.close()
                except:
                    pass
            return False
    
    def communication_loop(self):
        """Handle ongoing communication with the PC"""
        
        try:
            while self.running and self.connected:
                try:
                    # Wait for command from PC
                    self.socket.settimeout(10.0)
                    data = self.socket.recv(1024)
                    
                    if not data:
                        print("Connection closed by PC")
                        break
                    
                    # Process command
                    command = data.decode()
                    print("Received:", command)
                    
                    # Send response
                    response = "DATA_EV3_" + str(EV3_ID) + "_" + str(self.data_counter)
                    self.data_counter += 1
                    self.socket.send(response.encode())
                    print("Sent:", response)
                    
                except bluetooth.BluetoothError as be:
                    print("Bluetooth error:", be)
                    break
                except Exception as e:
                    print("Communication error:", e)
                    break
                    
        except Exception as e:
            print("Communication loop error:", e)
        finally:
            # Clean up
            self.disconnect()
    
    def disconnect(self):
        """Clean up connection resources"""
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.connected = False
        print("Disconnected from PC")
    
    def stop(self):
        self.running = False
        self.disconnect()
        print("Client stopped")

def main():
    client = EV3Client()
    
    if not client.start():
        print("Failed to start client")
        return
    
    try:
        # Keep main thread running and show status
        print("EV3", EV3_ID, "running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            print("EV3", EV3_ID, "client", "connected" if client.connected else "disconnected")
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        client.stop()

if __name__ == "__main__":
    main()