#!/usr/bin/env python3

import bluetooth
import threading
import time
import os
import signal
import subprocess

# EV3 ID - set this differently for each EV3 (1, 2, or 3)
EV3_ID = 1

# RFCOMM port
PORT = 5

class EV3Server:
    def __init__(self):
        self.server_socket = None
        self.running = False
        self.data_counter = 0
    
    def reset_bluetooth(self):
        """Reset Bluetooth service to clear any stale connections"""
        try:
            print("Resetting Bluetooth service...")
            # Try to restart Bluetooth service - may require sudo
            try:
                subprocess.run(["systemctl", "restart", "bluetooth.service"], timeout=5)
                time.sleep(2)  # Wait for service to restart
            except:
                print("Could not restart Bluetooth service, continuing anyway")
            
            # Kill any existing rfcomm processes
            try:
                subprocess.run(["pkill", "-f", "rfcomm"], timeout=5)
            except:
                pass
                
            time.sleep(1)  # Wait for processes to terminate
            print("Bluetooth reset complete")
            return True
        except Exception as e:
            print("Error resetting Bluetooth:", e)
            return False
    
    def start(self):
        # Reset Bluetooth before starting server
        self.reset_bluetooth()
        
        # Try to clear the port if it's in use
        try:
            dummy_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            dummy_socket.bind(("", PORT))
            dummy_socket.close()
        except:
            pass
        
        try:
            # Create server socket
            self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            
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
                self.server_socket.settimeout(None)  # No timeout for accept - blocking call
                
                client_sock, client_info = self.server_socket.accept()
                client_address = client_info[0]
                
                print("PC connected from:", client_address)
                
                # Handle this connection
                self.handle_connection(client_sock)
                
                # Brief pause between connections
                time.sleep(0.5)
                
            except bluetooth.BluetoothError as be:
                if self.running:  # Only report if we're still supposed to be running
                    print("Bluetooth error accepting connection:", be)
                    # Try to recover by closing and reopening server socket
                    try:
                        self.server_socket.close()
                        time.sleep(1)
                        self.server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                        self.server_socket.bind(("", PORT))
                        self.server_socket.listen(1)
                        print("Recovered server socket after accept error")
                    except Exception as e:
                        print("Failed to recover server socket:", e)
                        # Try to restart the entire server
                        self.stop()
                        time.sleep(1)
                        self.start()
                        break
                time.sleep(1)
            except Exception as e:
                if self.running:
                    print("Error accepting connection:", e)
                time.sleep(1)
    
    def handle_connection(self, sock):
        try:
            # Set timeout for client operations
            sock.settimeout(5.0)
            
            # Receive initial message
            try:
                data = sock.recv(1024)
                if not data:
                    print("No data received in initial exchange")
                    return
                
                message = data.decode()
                print("Received:", message)
                
                # Send response with current data
                response = "DATA_EV3_" + str(EV3_ID) + "_" + str(self.data_counter)
                self.data_counter += 1
                sock.send(response.encode())
                print("Sent:", response)
                
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
                
                # Process command (example: extract command number)
                parts = command.split("_")
                if len(parts) > 1 and parts[0] == "CMD":
                    cmd_num = parts[1]
                    
                    # Send acknowledgment
                    ack = "ACK_EV3_" + str(EV3_ID) + "_" + cmd_num
                    sock.send(ack.encode())
                    print("Sent acknowledgment:", ack)
            except Exception as e:
                print("Error processing command:", e)
                
        except Exception as e:
            print("Connection handling error:", e)
        finally:
            # Close connection
            try:
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