#!/usr/bin/env python3

import bluetooth
import threading
import time
import sys

# Configuration
PORT = 1
EV3_ID = 2  # Change this for each EV3 (1, 2, or 3)

class BluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None
        self.socket_lock = threading.Lock()

    def setup_server(self):
        try:
            # Create new server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Bind to port - don't use socket options that aren't available
            self.server_socket.bind(("", PORT))
            
            # Listen for connections
            self.server_socket.listen(1)
            
            print("EV3 # waiting for connection on RFCOMM channel ", EV3_ID, PORT)
            return True
        except Exception as e:
            print("Server setup failed: ", e)
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
            return False

    def accept_connection(self):
        if not self.server_socket:
            return False
            
        try:
            print("Waiting for connection...")
            # Accept connection
            self.client_socket, client_info = self.server_socket.accept()
            
            print("Accepted connection from ", client_info)
            
            # Mark as connected
            self.connected = True
            
            # Allow connection to stabilize
            time.sleep(1.0)
            
            # Send ready message
            if not self.send_data("READY_EV3_"+str(EV3_ID)):
                print("Failed to send ready message")
                self.close_client()
                return False
            
            # Start receive thread
            self.stop_thread = False
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            print("Failed to accept connection: ", e)
            self.close_client()
            return False

    def send_data(self, data):
        if not self.connected or not self.client_socket:
            return False
            
        with self.socket_lock:
            try:
                # Convert to bytes if needed
                if isinstance(data, str):
                    data = data.encode()
                    
                # Send data
                self.client_socket.send(data)
                
                # Small delay after sending
                time.sleep(0.05)
                return True
            except Exception as e:
                # print(f"Error sending data: {e}")
                print("Error sending data: ", e)
                self.connected = False
                return False

    def receive_data(self):
        buffer_size = 1024
        
        while not self.stop_thread and self.connected:
            with self.socket_lock:
                if not self.client_socket:
                    self.connected = False
                    break
                    
                try:
                    # Try to receive data - don't set timeout as it might not be supported
                    data = self.client_socket.recv(buffer_size)
                    
                    # Process received data
                    if data:
                        try:
                            message = data.decode('utf-8')
                            print("Received: ", message)
                            
                            # Send acknowledgment
                            # ack_message = f"ACK_EV3_{EV3_ID}_{message.split('_')[-1] if '_' in message else '0'}"
                            ack_message = "ACK_EV3_"+str(EV3_ID)+"_" + (message.split('_')[-1] if '_' in message else '0')
                            self.send_data(ack_message)
                        except UnicodeDecodeError:
                            print("Received binary data")
                    else:
                        # Empty data means connection closed
                        print("Connection closed by remote device")
                        self.connected = False
                        break
                        
                except Exception as e:
                    # print(f"Error receiving data: {e}")
                    print("Error receiving data: ", e)
                    self.connected = False
                    break
            
            # Sleep outside the lock
            time.sleep(0.1)

    def close_client(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        self.connected = False

    def close(self):
        self.stop_thread = True
        
        # Wait for thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(1.0)
        
        # Close client socket
        self.close_client()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
            
        print("Bluetooth server closed")

def main():
    retry_count = 0
    max_retries = 10  # More retries
    retry_delay = 5   # Seconds between retries
    
    print("Starting EV3 # Bluetooth server", EV3_ID)
    
    try:
        while retry_count < max_retries:
            server = BluetoothServer()
            
            if not server.setup_server():
                print("Failed to set up Bluetooth server. Retry ", retry_count+1/max_retries)
                retry_count += 1
                time.sleep(retry_delay)
                continue
            
            if not server.accept_connection():
                print("Failed to accept connection. Retry ", retry_count+1/max_retries)
                server.close()
                retry_count += 1
                time.sleep(retry_delay)
                continue
            
            # Reset retry count on successful connection
            retry_count = 0
            
            # Data counter
            counter = 0
            
            # Main communication loop
            try:
                while server.connected:
                    # Send periodic data with unique counter
                    # message = f"DATA_EV3_{EV3_ID}_{counter}"
                    message = "DATA_EV3"+"_"+str(EV3_ID)+"_"+str(counter)
                    
                    if not server.send_data(message):
                        print("Failed to send data, connection lost")
                        break
                        
                    counter += 1
                    time.sleep(0.3)  # 300ms pause between sends
            except Exception as e:
                # print(f"Error in main loop: {e}")
                print("Error in main loop: ", e)
                
            print("Connection closed, cleaning up...")
            server.close()
            time.sleep(retry_delay)
            
        # print(f"Maximum retries ({max_retries}) reached. Exiting.")
        print("Maximum retries () reached. Exiting.", max_retries)
        
    except KeyboardInterrupt:
        print("Program terminated by user")
        if 'server' in locals():
            server.close()
        
if __name__ == "__main__":
    main()