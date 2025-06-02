#!/usr/bin/env python3

import bluetooth
import threading
import time
import socket

# RFCOMM port
PORT = 5

class PCServer:
    def __init__(self):
        self.server_socket = None
        self.running = False
        self.command_counter = 0
        self.connected_clients = {}
        self.lock = threading.Lock()
    
    def start(self):
        try:
            # Create server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Try to set socket options, but continue if it fails
            try:
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception as e:
                print("Note: Could not set socket options:", e)
            
            # Bind to port - use empty string for any available adapter
            print("Binding to port", PORT)
            self.server_socket.bind(("", PORT))
            self.server_socket.listen(5)  # Listen for multiple EV3 connections
            
            # Try to get server address
            try:
                server_address = bluetooth.read_local_bdaddr()[0]
                print("PC server started on address:", server_address, "port:", PORT)
            except:
                print("PC server started on port:", PORT, "(could not determine address)")
            
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
                print("Waiting for EV3 to connect...")
                self.server_socket.settimeout(10.0)  # 10 second timeout for accept
                
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("EV3 connected from:", client_address)
                
                # Create a handling thread for this connection
                handler_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_sock, client_address)
                )
                handler_thread.daemon = True
                handler_thread.start()
                
            except bluetooth.BluetoothError as be:
                if self.running:  # Only report if we're still supposed to be running
                    print("Bluetooth error accepting connection:", be)
                time.sleep(0.5)
            except Exception as e:
                if self.running:
                    print("Error accepting connection:", e)
                time.sleep(0.5)
    
    def handle_client(self, sock, address):
        """Handle communication with a connected EV3"""
        ev3_id = None
        
        try:
            # Set timeout for operations
            sock.settimeout(2.0)
            
            # Wait for initial message to identify which EV3 this is
            try:
                data = sock.recv(1024)
                if not data:
                    print("No identification received from EV3")
                    return
                
                message = data.decode()
                print("Received from EV3:", message)
                
                # Extract EV3 ID from the message (assuming format like "HELLO_EV3_1")
                parts = message.split("_")
                if len(parts) > 2 and parts[0] == "HELLO" and parts[1] == "EV3":
                    try:
                        ev3_id = int(parts[2])
                        print("Identified as EV3", ev3_id)
                        
                        # Store the connection
                        with self.lock:
                            self.connected_clients[ev3_id] = {
                                "socket": sock,
                                "address": address,
                                "last_seen": time.time()
                            }
                    except:
                        print("Could not parse EV3 ID from message:", message)
                
                # Send response
                response = "HELLO_PC"
                sock.send(response.encode())
                print("Sent to EV3:", response)
                
            except Exception as e:
                print("Error in initial exchange:", e)
                return
            
            # Main communication loop with this EV3
            while self.running:
                try:
                    # Send a command periodically
                    with self.lock:
                        command = "CMD_" + str(self.command_counter)
                        self.command_counter += 1
                    
                    sock.send(command.encode())
                    print("Sent command to EV3", ev3_id, ":", command)
                    
                    # Wait for response
                    sock.settimeout(1.0)
                    data = sock.recv(1024)
                    if data:
                        response = data.decode()
                        print("Received from EV3", ev3_id, ":", response)
                        
                        # Update last seen time
                        if ev3_id in self.connected_clients:
                            self.connected_clients[ev3_id]["last_seen"] = time.time()
                    
                    # Wait before next command
                    time.sleep(0.5)  # This controls the communication rate
                    
                except bluetooth.BluetoothError as be:
                    print("Bluetooth error with EV3", ev3_id, ":", be)
                    break
                except Exception as e:
                    print("Error communicating with EV3", ev3_id, ":", e)
                    break
                    
        except Exception as e:
            print("Connection handling error:", e)
        finally:
            # Remove from connected clients
            if ev3_id is not None and ev3_id in self.connected_clients:
                with self.lock:
                    del self.connected_clients[ev3_id]
            
            # Close connection
            try:
                sock.close()
            except:
                pass
            print("Disconnected from EV3", ev3_id)
    
    def stop(self):
        self.running = False
        
        # Close all client connections
        with self.lock:
            for ev3_id, client in list(self.connected_clients.items()):
                try:
                    client["socket"].close()
                except:
                    pass
                print("Closed connection to EV3", ev3_id)
            self.connected_clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("Server stopped")

def main():
    server = PCServer()
    
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