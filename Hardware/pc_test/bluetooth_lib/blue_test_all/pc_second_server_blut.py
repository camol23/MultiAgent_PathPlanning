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
            self.server_socket.bind(("", port))
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
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("Accepted connection from:", client_address)
                
                # Start a thread to handle this client
                client_thread = threading.Thread(target=self.handle_client, 
                                              args=(client_sock, client_address))
                client_thread.daemon = True
                client_thread.start()
                
                # Store the client socket
                with self.lock:
                    self.client_sockets[client_address] = client_sock
                
            except Exception as e:
                print("Error accepting connection:", e)
                time.sleep(1)
    
    def handle_client(self, client_sock, address):
        try:
            while self.running:
                # Try to receive data
                data = client_sock.recv(1024)
                
                if not data:
                    # Connection closed
                    print("Connection closed by EV3 at:", address)
                    break
                
                # Process received data
                message = data.decode()
                print("Received from", address, ":", message)
                
                # Wait a small amount before next receive
                time.sleep(0.1)
                
        except Exception as e:
            print("Error handling client", address, ":", e)
        finally:
            # Clean up client connection
            try:
                client_sock.close()
            except:
                pass
            
            # Remove from clients dictionary
            with self.lock:
                if address in self.client_sockets:
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
            time.sleep(0.3)
    
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
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        server.stop()

if __name__ == "__main__":
    main()