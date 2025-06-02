#!/usr/bin/env python3

import bluetooth
import threading
import time

# RFCOMM port
port = 5
MAX_CLIENTS = 3  # Maximum number of EV3 clients

class BluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_sockets = {}  # Dictionary to store client connections by address
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        try:
            # Create server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Close and rebind to ensure we're not reusing a stale socket
            try:
                old_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                old_socket.bind(("", port))
                old_socket.close()
                print("Closed any existing socket")
                time.sleep(1)
            except:
                pass
            
            # Bind to port
            print("Binding to port", port)
            self.server_socket.bind(("", port))
            self.server_socket.listen(MAX_CLIENTS)
            
            print("PC server started on RFCOMM port", port)
            print("Waiting for up to", MAX_CLIENTS, "connections")
            
            self.running = True
            
            # Start thread to accept connections
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # Start thread to send periodic messages
            send_thread = threading.Thread(target=self.send_periodic_messages)
            send_thread.daemon = True
            send_thread.start()
            
            return True
        except Exception as e:
            print("Error starting server:", e)
            return False
    
    def accept_connections(self):
        while self.running:
            try:
                print("Waiting for a new connection...")
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("Accepted connection from:", client_address)
                
                # Check if we already have this client
                with self.lock:
                    if client_address in self.client_sockets:
                        # Close existing connection
                        print("Closing existing connection from:", client_address)
                        try:
                            self.client_sockets[client_address].close()
                        except:
                            pass
                    
                    # Store new connection
                    self.client_sockets[client_address] = client_sock
                
                # Start a thread to handle this client
                client_thread = threading.Thread(target=self.handle_client, 
                                              args=(client_sock, client_address))
                client_thread.daemon = True
                client_thread.start()
                
                # Print active connections
                with self.lock:
                    print("Active connections:", len(self.client_sockets))
                    for addr in self.client_sockets.keys():
                        print("- Connected:", addr)
                
            except Exception as e:
                print("Error in accept loop:", e)
                time.sleep(1)
    
    def handle_client(self, client_sock, address):
        ev3_id = None
        try:
            # Receive initial message (should contain EV3_ID)
            try:
                data = client_sock.recv(1024)
                if data:
                    message = data.decode()
                    print("Initial message from", address, ":", message)
                    
                    # Extract EV3_ID if in format "HELLO_EV3_X"
                    parts = message.split("_")
                    if len(parts) > 2 and parts[0] == "HELLO" and parts[1] == "EV3":
                        ev3_id = parts[2]
                        print("Identified as EV3", ev3_id)
                    
                    # Send welcome message
                    welcome_msg = "WELCOME_" + str(ev3_id)
                    client_sock.send(welcome_msg.encode())
                    print("Sent welcome to EV3", ev3_id)
            except Exception as e:
                print("Error in initial message exchange:", e)
                return
            
            # Main communication loop
            while self.running:
                try:
                    # Try to receive data
                    data = client_sock.recv(1024)
                    
                    if not data:
                        # Connection closed
                        print("Connection closed by EV3", ev3_id)
                        break
                    
                    # Process received data
                    message = data.decode()
                    print("Received from EV3", ev3_id, ":", message)
                    
                except Exception as e:
                    print("Error handling EV3", ev3_id, ":", e)
                    break
                    
                time.sleep(0.05)  # Small wait to prevent CPU hogging
                
        except Exception as e:
            print("Fatal error handling EV3", ev3_id, ":", e)
        finally:
            # Clean up client connection
            with self.lock:
                if address in self.client_sockets:
                    try:
                        self.client_sockets[address].close()
                    except:
                        pass
                    del self.client_sockets[address]
            
            print("Disconnected EV3", ev3_id)
    
    def send_periodic_messages(self):
        counter = 0
        
        while self.running:
            # Sleep at the beginning to allow initial connections
            time.sleep(0.5)
            
            # Create message
            message = "CMD_" + str(counter)
            
            # Get copy of client sockets to avoid modification during iteration
            with self.lock:
                clients = list(self.client_sockets.items())
            
            # Send to all connected clients
            for address, sock in clients:
                try:
                    sock.send(message.encode())
                    print("Sent to", address, ":", message)
                except Exception as e:
                    print("Error sending to", address, ":", e)
            
            # Increment counter
            counter += 1
    
    def stop(self):
        self.running = False
        
        # Close all client connections
        with self.lock:
            for sock in self.client_sockets.values():
                try:
                    sock.close()
                except:
                    pass
            self.client_sockets.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("Server stopped")

def main():
    server = BluetoothServer()
    
    if not server.start():
        print("Failed to start server")
        return
    
    try:
        # Keep main thread running
        print("Server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        server.stop()

if __name__ == "__main__":
    main()