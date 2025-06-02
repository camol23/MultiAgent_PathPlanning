#!/usr/bin/env python3

import bluetooth
import time
import threading

# PC server address and port
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 5

# EV3 ID - set differently for each EV3 (1, 2, or 3)
EV3_ID = 1

class EV3BluetoothClient:
    def __init__(self):
        self.sock = None
        self.connected = False
        self.running = False
        self.data_counter = 0
        self.send_lock = threading.Lock()
    
    def connect(self):
        """Connect to PC server"""
        try:
            # Create socket
            self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.sock.settimeout(10.0)  # 10 second connection timeout
            
            # Connect to PC server
            print("Connecting to PC server at", PC_MAC_ADDRESS)
            self.sock.connect((PC_MAC_ADDRESS, PORT))
            print("Connected to PC server")
            
            # Send initial identification message
            hello_msg = "HELLO_EV3_" + str(EV3_ID)
            self.sock.send(hello_msg.encode())
            print("Sent identification:", hello_msg)
            
            self.connected = True
            return True
        except Exception as e:
            print("Connection failed:", e)
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            return False
    
    def start(self):
        """Start the client operations"""
        self.running = True
        
        # Start data sending thread
        send_thread = threading.Thread(target=self.send_data_loop)
        send_thread.daemon = True
        send_thread.start()
        
        # Start receive thread
        receive_thread = threading.Thread(target=self.receive_loop)
        receive_thread.daemon = True
        receive_thread.start()
        
        return True
    
    def send_data_loop(self):
        """Loop to periodically send data to PC"""
        while self.running:
            if not self.connected:
                time.sleep(0.5)
                continue
                
            try:
                with self.send_lock:
                    # Create data message
                    data_msg = "DATA_EV3_" + str(EV3_ID) + "_" + str(self.data_counter)
                    self.data_counter += 1
                    
                    # Send data
                    self.sock.send(data_msg.encode())
                    print("Sent data:", data_msg)
            except Exception as e:
                print("Error sending data:", e)
                self.handle_connection_loss()
                
            # Wait before sending next data
            time.sleep(0.3)  # 300ms between sends
    
    def receive_loop(self):
        """Loop to receive commands from PC"""
        while self.running:
            if not self.connected:
                time.sleep(0.5)
                continue
                
            try:
                # Try to receive with a reasonable timeout
                self.sock.settimeout(1.0)
                data = self.sock.recv(1024)
                
                if not data:
                    print("Connection closed by PC")
                    self.handle_connection_loss()
                    continue
                    
                # Process command
                command = data.decode()
                print("Received command:", command)
                
                # Parse command
                parts = command.split("_")
                if len(parts) > 1 and parts[0] == "CMD":
                    cmd_num = parts[1]
                    
                    # Send acknowledgment
                    with self.send_lock:
                        ack = "ACK_EV3_" + str(EV3_ID) + "_" + cmd_num
                        self.sock.send(ack.encode())
                        print("Sent acknowledgment:", ack)
                
            except bluetooth.btcommon.BluetoothError as be:
                # Ignore timeout errors, which are expected
                if "timed out" not in str(be):
                    print("Bluetooth error:", be)
                    self.handle_connection_loss()
            except Exception as e:
                print("Error receiving command:", e)
                self.handle_connection_loss()
    
    def handle_connection_loss(self):
        """Handle loss of connection to PC"""
        if not self.connected:
            return
            
        self.connected = False
        
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        self.sock = None
        
        print("Connection to PC lost. Will try to reconnect...")
        
        # Start reconnection thread
        reconnect_thread = threading.Thread(target=self.reconnect_loop)
        reconnect_thread.daemon = True
        reconnect_thread.start()
    
    def reconnect_loop(self):
        """Loop to try reconnecting to PC server"""
        retry_delay = 5  # Seconds between reconnection attempts
        
        while self.running and not self.connected:
            print("Attempting to reconnect...")
            if self.connect():
                print("Reconnected to PC")
                return
            
            time.sleep(retry_delay)
    
    def stop(self):
        """Stop client operations"""
        self.running = False
        
        # Close connection
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        
        self.connected = False
        print("Client stopped")

def main():
    client = EV3BluetoothClient()
    
    # Initial connection attempt
    connect_success = client.connect()
    if not connect_success:
        print("Initial connection failed. Will retry in background.")
    
    # Start client operations regardless
    client.start()
    
    try:
        # Keep main thread running and show status
        print("EV3", EV3_ID, "client running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(10)
            if client.connected:
                print("EV3", EV3_ID, "still connected to PC.")
            else:
                print("EV3", EV3_ID, "trying to reconnect to PC...")
                
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        client.stop()

if __name__ == "__main__":
    main()