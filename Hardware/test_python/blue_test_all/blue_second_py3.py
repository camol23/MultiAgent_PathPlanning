#!/usr/bin/env python3

import bluetooth
import threading
import time

# Configuration
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 1
EV3_ID = 1  # Change this for each EV3 (1, 2, or 3)

class BluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None

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
            self.connected = True
            print("Accepted connection from", client_info)
            
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
        if self.connected:
            try:
                self.client_socket.send(data.encode())
                return True
            except Exception as e:
                print("Error sending data:", str(e))
                self.connected = False
                return False
        return False

    def receive_data(self):
        while not self.stop_thread:
            if self.connected:
                try:
                    data = self.client_socket.recv(1024).decode()
                    if data:
                        print("Received:", data)
                        # Process the command here if needed
                        # For example, respond with acknowledgment
                        self.send_data("ACK_EV3_" + str(EV3_ID))
                except Exception as e:
                    print("Error receiving data:", str(e))
                    self.connected = False
                    break
            time.sleep(0.1)  # 100ms pause

    def close(self):
        self.stop_thread = True
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        self.connected = False
        print("Bluetooth server closed")

def main():
    server = BluetoothServer()
    
    if not server.setup_server():
        print("Failed to set up Bluetooth server. Exiting.")
        return
    
    try:
        # Wait for connection
        if server.accept_connection():
            counter = 0
            
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
            server.close()
            main()  # Restart server
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        server.close()

if __name__ == "__main__":
    main()