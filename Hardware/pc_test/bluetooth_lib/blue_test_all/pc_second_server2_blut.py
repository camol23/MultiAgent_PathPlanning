#!/usr/bin/env python3

import bluetooth
import threading
import time

# RFCOMM port
PORT = 5

class PCBluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_sockets = {}  # Dictionary to store client connections by address
        self.client_ids = {}      # Map addresses to EV3 IDs once known
        self.running = False
        self.lock = threading.Lock()
        self.command_counter = 0
    
    def start(self):
        try:
            # Create server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
            # Try to set socket options for better reliability
            try:
                self.server_socket.setsockopt(bluetooth.SOL_SOCKET, bluetooth.SO_REUSEADDR, 1)
            except:
                pass  # Continue if this fails
                
            self.server_socket.bind(("", PORT))
            self.server_socket.listen(3)  # Allow up to 3 EV3 connections
            
            print("PC server started on RFCOMM port", PORT)
            
            self.running = True
            
            # Start thread to accept connections
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # Start thread to send commands to all clients
            command_thread = threading.Thread(target=self.send_commands_to_clients)
            command_thread.daemon = True
            command_thread.start()
            
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
                # Wait for client connections
                print("Waiting for EV3 connections...")
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
        ev3_id = None
        
        try:
            # Set a reasonable timeout
            client_sock.settimeout(5.0)
            
            # Wait for initial hello message to identify EV3
            data = client_sock.recv(1024)
            if data:
                message = data.decode()
                print("Initial message from", address, ":", message)
                
                # Try to parse EV3 ID from message
                parts = message.split('_')
                if len(parts) >= 3 and parts[0] == "HELLO" and parts[1] == "EV3":
                    try:
                        ev3_id = int(parts[2])
                        print("Identified as EV3", ev3_id)
                        
                        # Store the mapping
                        with self.lock:
                            self.client_ids[address] = ev3_id
                    except:
                        print("Could not parse EV3 ID from message")
            
            # Main communication loop
            while self.running:
                try:
                    # Try to receive data with a short timeout
                    client_sock.settimeout(0.5)
                    data = client_sock.recv(1024)
                    
                    if not data:
                        print("Connection closed by EV3 at:", address)
                        break
                    
                    # Process received data
                    message = data.decode()
                    print("Received from", address, ":", message)
                    
                except bluetooth.btcommon.BluetoothError as be:
                    # Ignore timeout errors, which are expected
                    if "timed out" not in str(be):
                        print("Bluetooth error with", address, ":", be)
                        break
                except Exception as e:
                    print("Error receiving from", address, ":", e)
                    break
                
                # Small delay to prevent tight loop
                time.sleep(0.05)
                
        except Exception as e:
            print("Error handling client", address, ":", e)
        finally:
            # Clean up client connection
            try:
                client_sock.close()
            except:
                pass
            
            # Remove from dictionaries
            with self.lock:
                if address in self.client_sockets:
                    del self.client_sockets[address]
                if address in self.client_ids:
                    del self.client_ids[address]
            
            if ev3_id:
                print("Disconnected EV3", ev3_id)
            else:
                print("Disconnected client:", address)
    
    def send_commands_to_clients(self):
        """Send periodic commands to all connected clients"""
        while self.running:
            # Only proceed if we have clients
            with self.lock:
                if not self.client_sockets:
                    time.sleep(0.1)
                    continue
                
                # Create command with sequential counter
                command = "CMD_" + str(self.command_counter)
                self.command_counter += 1
                
                # Copy client data to avoid holding lock during sends
                clients = list(self.client_sockets.items())
                ids = dict(self.client_ids)
            
            # Send to each connected client
            for address, sock in clients:
                ev3_id = ids.get(address, "unknown")
                try:
                    sock.send(command.encode())
                    print("Sent to EV3", ev3_id, ":", command)
                except Exception as e:
                    print("Error sending to EV3", ev3_id, ":", e)
                    # Don't close here - let the client handler thread detect the error
                
                # Small delay between sends to different clients
                time.sleep(0.05)
            
            # Wait before next command cycle - adjust as needed
            # This controls how often commands are sent
            time.sleep(0.2)  # 200ms between command cycles
    
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
            self.client_ids.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("Server stopped")

def main():
    server = PCBluetoothServer()
    
    if not server.start():
        print("Failed to start server")
        return
    
    try:
        # Keep main thread running and show status
        print("PC server running. Press Ctrl+C to stop.")
        client_count = 0
        while True:
            time.sleep(5)
            with server.lock:
                new_count = len(server.client_sockets)
                if new_count != client_count:
                    client_count = new_count
                    print("Currently connected EV3s:", client_count)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        server.stop()

if __name__ == "__main__":
    main()