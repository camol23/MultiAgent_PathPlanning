#!/usr/bin/env python3


# ERROR : Error starting server: module 'bluetooth' has no attribute 'SOL_SOCKET'

import bluetooth
import threading
import time

# EV3 ID - set this differently for each EV3 (1, 2, or 3)
EV3_ID = 1

# RFCOMM port
PORT = 5

class EV3Server:
    def __init__(self):
        self.server_socket = None
        self.running = False
        self.data_counter = 0
    
    def start(self):
        try:
            # Create server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.server_socket.setsockopt(bluetooth.SOL_SOCKET, bluetooth.SO_REUSEADDR, 1)
            
            # Bind to port
            print("Binding to port", PORT)
            self.server_socket.bind(("", PORT))
            self.server_socket.listen(1)  # Only need to listen for one connection (PC)
            
            print("EV3", EV3_ID, "server started on port", PORT)
            
            self.running = True
            
            # Main server loop in a separate thread
            server_thread = threading.Thread(target=self.accept_connections)
            server_thread.daemon = True
            server_thread.start()
            
            return True
        except Exception as e:
            print("Error starting server:", e)
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
            return False
    
    def accept_connections(self):
        while self.running:
            try:
                print("Waiting for PC to connect...")
                self.server_socket.settimeout(10.0)  # 10 second timeout for accept
                
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("PC connected from:", client_address)
                
                # Handle this connection
                self.handle_connection(client_sock)
                
            except bluetooth.BluetoothError as be:
                if self.running:  # Only report if we're still supposed to be running
                    print("Bluetooth error accepting connection:", be)
                time.sleep(0.5)
            except Exception as e:
                if self.running:
                    print("Error accepting connection:", e)
                time.sleep(0.5)
    
    def handle_connection(self, sock):
        try:
            # Set timeout for client operations
            sock.settimeout(2.0)
            
            # Receive initial message
            try:
                data = sock.recv(1024)
                if not data:
                    print("No data received in initial exchange")
                    return
                
                message = data.decode()
                print("Received:", message)
                
                # Add small delay for stability
                time.sleep(0.1)
                
                # Send response with current data
                response = "DATA_EV3_" + str(EV3_ID) + "_" + str(self.data_counter)
                self.data_counter += 1
                sock.send(response.encode())
                print("Sent:", response)
                
                # Add small delay between messages
                time.sleep(0.1)
                
            except bluetooth.BluetoothError as be:
                print("Bluetooth error in initial exchange:", be)
                return
            except Exception as e:
                print("Error in initial exchange:", e)
                return
            
            # Receive command
            try:
                data = sock.recv(1024)
                if not data:
                    print("No command data received")
                    return
                
                command = data.decode()
                print("Received command:", command)
                
                # Add small delay for stability
                time.sleep(0.1)
                
                # Process command (example: extract command number)
                parts = command.split("_")
                if len(parts) > 1 and parts[0] == "CMD":
                    cmd_num = parts[1]
                    
                    # Send acknowledgment
                    ack = "ACK_EV3_" + str(EV3_ID) + "_" + cmd_num
                    sock.send(ack.encode())
                    print("Sent acknowledgment:", ack)
            except bluetooth.BluetoothError as be:
                print("Bluetooth error processing command:", be)
            except Exception as e:
                print("Error processing command:", e)
                
        except Exception as e:
            print("Connection handling error:", e)
        finally:
            # Close connection
            try:
                time.sleep(0.1)  # Wait a moment before closing
                sock.shutdown(2)
                sock.close()
            except:
                pass
            print("Disconnected from PC")
    
    def stop(self):
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("Server stopped")

def main():
    server = EV3Server()
    
    if not server.start():
        print("Failed to start server")
        return
    
    try:
        # Keep main thread running and show status
        print("EV3", EV3_ID, "running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            print("EV3", EV3_ID, "server still running...")
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        server.stop()

if __name__ == "__main__":
    main()