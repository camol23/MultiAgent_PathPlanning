#!/usr/bin/env python3

import bluetooth
import threading
import time

# RFCOMM port
port = 5

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
            
            # Bind to port, with retry if needed
            try_count = 0
            while try_count < 5:
                try:
                    self.server_socket.bind(("", port))
                    break
                except bluetooth.btcommon.BluetoothError as e:
                    print("Bind attempt failed:", e)
                    try_count += 1
                    time.sleep(2)
                    
            if try_count == 5:
                print("Failed to bind to port after multiple attempts")
                return False
                
            self.server_socket.listen(5)  # Allow up to 5 connections
            
            print("PC server started on RFCOMM port", port)
            
            self.running = True
            
            # Start thread to accept connections
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # Start thread to send periodic messages to all clients
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
                # Don't use timeout for server socket - can cause issues
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("Accepted connection from:", client_address)
                
                # Store the client socket first to prevent race conditions
                with self.lock:
                    if client_address in self.client_sockets:
                        # Close old connection if it exists
                        try:
                            old_sock = self.client_sockets[client_address]
                            old_sock.close()
                            print("Closed existing connection from:", client_address)
                        except:
                            pass
                    self.client_sockets[client_address] = client_sock
                
                # Start a thread to handle this client
                client_thread = threading.Thread(target=self.handle_client, 
                                              args=(client_sock, client_address))
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                print("Error in accept loop:", e)
                time.sleep(1)
    
    def handle_client(self, client_sock, address):
        try:
            # First data exchange - wait for initial message
            try:
                # Receive initial data with longer timeout
                data = client_sock.recv(1024)
                if data:
                    message = data.decode()
                    print("Initial message from", address, ":", message)
                    
                    # Send immediate welcome response
                    welcome = "WELCOME_" + address.replace(":", "")
                    try:
                        client_sock.send(welcome.encode())
                        print("Sent welcome to", address, ":", welcome)
                    except Exception as e:
                        print("Error sending welcome:", e)
            except Exception as e:
                print("Error receiving initial data:", e)
                return
            
            # Main communication loop
            while self.running:
                try:
                    # Try to receive data
                    data = client_sock.recv(1024)
                    
                    if not data:
                        # Connection closed
                        print("Connection closed by client at:", address)
                        break
                    
                    # Process received data
                    message = data.decode()
                    print("Received from", address, ":", message)
                    
                except bluetooth.btcommon.BluetoothError as e:
                    if "timed out" not in str(e).lower():
                        print("Bluetooth error with client", address, ":", e)
                        break
                except Exception as e:
                    print("Error handling client", address, ":", e)
                    break
                    
                # Small wait to prevent CPU hogging
                time.sleep(0.05)
                
        except Exception as e:
            print("Fatal error handling client", address, ":", e)
        finally:
            # Clean up client connection
            with self.lock:
                if address in self.client_sockets:
                    try:
                        self.client_sockets[address].close()
                    except:
                        pass
                    del self.client_sockets[address]
            
            print("Disconnected client:", address)
    
    def send_periodic_messages(self):
        counter = 0
        
        while self.running:
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
                    # Don't remove here, let the client handler thread do it
            
            # Increment counter
            counter += 1
            
            # Wait before next send
            time.sleep(0.5)  # Slightly longer interval
    
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