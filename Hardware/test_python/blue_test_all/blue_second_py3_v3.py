#!/usr/bin/env python3

import bluetooth
import threading
import time

# Configuration
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 1
EV3_ID = 2  # Change this for each EV3 (1, 2, or 3)

class BluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None
        # Create a lock for socket operations
        self.socket_lock = threading.Lock()

    def setup_server(self):
        try:
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.server_socket.bind(("", PORT))
            self.server_socket.listen(1)
            print("Waiting for connection on RFCOMM channel", PORT)
            return True
        except Exception as e:
            print("Server setup failed:", str(e))
            return False

    def accept_connection(self):
        try:
            self.client_socket, client_info = self.server_socket.accept()
            
            # Add a small delay to stabilize the connection
            time.sleep(0.5)
            
            self.connected = True
            print("Accepted connection from", client_info)
            
            # Send initial READY message to PC
            with self.socket_lock:
                self.client_socket.send(("READY_EV3_" + str(EV3_ID)).encode())
            
            # Start receive thread
            self.stop_thread = False
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            return True
        except Exception as e:
            print("Failed to accept connection:", str(e))
            return False

    def send_data(self, data):
        if not self.connected:
            return False
            
        # Acquire lock before socket operations
        with self.socket_lock:
            try:
                self.client_socket.send(data.encode())
                return True
            except Exception as e:
                print("Error sending data:", str(e))
                self.connected = False
                return False

    def receive_data(self):
        while not self.stop_thread:
            if self.connected:
                # Acquire lock before socket operations
                if self.socket_lock.acquire(timeout=1.0):  # 1 second timeout
                    try:
                        # Set a timeout for receiving to avoid blocking forever
                        self.client_socket.settimeout(0.5)
                        
                        try:
                            data = self.client_socket.recv(1024).decode()
                            # Reset timeout to blocking mode
                            self.client_socket.settimeout(None)
                            
                            if data:
                                print("Received:", data)
                                # Process the command here if needed
                                # Send acknowledgment (outside the lock)
                                self.socket_lock.release()  # Release lock before sending to avoid deadlock
                                self.send_data("ACK_EV3_" + str(EV3_ID))
                                continue  # Skip the release at the end as we've already released
                            else:
                                # Empty data indicates closed connection
                                print("Connection closed by PC")
                                self.connected = False
                        except bluetooth.btcommon.BluetoothError as e:
                            if e.args[0] == 'timed out':
                                # Timeout is normal - just continue
                                pass
                            else:
                                print("Error receiving data:", str(e))
                                self.connected = False
                    finally:
                        # Always release the lock if we haven't released it earlier
                        if self.socket_lock.locked():
                            self.socket_lock.release()
            
            # Small delay to avoid CPU hogging
            time.sleep(0.1)

    def close(self):
        self.stop_thread = True
        
        # Wait for receive thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        
        # Acquire lock before closing socket
        with self.socket_lock:
            if self.client_socket:
                try:
                    # Proper socket shutdown sequence
                    self.client_socket.shutdown(bluetooth.SHUT_RDWR)
                except:
                    pass
                finally:
                    try:
                        self.client_socket.close()
                    except:
                        pass
                    
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
                    
        self.connected = False
        print("Bluetooth server closed")

def main():
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        server = BluetoothServer()
        
        if not server.setup_server():
            print("Failed to set up Bluetooth server. Retry ", retry_count+1/max_retries)
            retry_count += 1
            time.sleep(5)
            continue
        
        try:
            # Wait for connection
            if server.accept_connection():
                counter = 0
                retry_count = 0  # Reset retry count on successful connection
                
                # Main loop - periodically send data
                while server.connected:
                    # Example: send periodic data
                    message = "DATA_EV3_" + str(EV3_ID) + "_" + str(counter)
                    if not server.send_data(message):
                        print("Connection lost")
                        break
                    
                    counter += 1
                    time.sleep(0.3)  # 300ms pause between sends
                    
                print("Connection closed. Waiting for new connection...")
            
        except KeyboardInterrupt:
            print("Program terminated by user")
            server.close()
            break
        finally:
            server.close()
            time.sleep(2)  # Wait before restarting server
            
    print("Maximum retries reached. Exiting.", max_retries)

if __name__ == "__main__":
    main()