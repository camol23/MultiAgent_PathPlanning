#!/usr/bin/env python3

import bluetooth
import threading
import time

# EV3 device MAC addresses (replace with your actual EV3 MAC addresses)
# ev3_addresses = [
#     "00:16:53:XX:XX:XX",  # EV3 #1
#     "00:16:53:YY:YY:YY",  # EV3 #2
#     "00:16:53:ZZ:ZZ:ZZ"   # EV3 #3
# ]

# F0:45:DA:11:92:74 one
# F0:45:DA:D2:C8:88 two (ticket)

ev3_addresses = [
    "F0:45:DA:11:92:74",  # EV3 #
    "F0:45:DA:D2:C8:88"
]

# RFCOMM port (must match on EV3s)
port = 1

class EV3Connection:
    def __init__(self, mac_address, ev3_id):
        self.mac_address = mac_address
        self.ev3_id = ev3_id
        self.socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None
        # Create a lock for socket operations
        self.socket_lock = threading.Lock()

    def connect(self):
        try:
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.connect((self.mac_address, port))
            
            # Add a small delay to stabilize the connection
            time.sleep(0.5)
            
            self.connected = True
            print("Connected to EV3 #" + str(self.ev3_id))
            
            # Start receive thread
            self.stop_thread = False
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            return True
        except Exception as e:
            print("Failed to connect to EV3 #" + str(self.ev3_id) + ": " + str(e))
            return False

    def send_data(self, data):
        if not self.connected:
            return False
            
        # Acquire lock before socket operations
        with self.socket_lock:
            try:
                # Check if socket is still valid
                try:
                    peer_name = self.socket.getpeername()
                except:
                    self.connected = False
                    print("Socket no longer valid for EV3 #" + str(self.ev3_id))
                    return False
                
                # Send data
                self.socket.send(data.encode())
                return True
            except Exception as e:
                print("Error sending to EV3 #" + str(self.ev3_id) + ": " + str(e))
                self.connected = False
                return False

    def receive_data(self):
        while not self.stop_thread:
            if self.connected:
                # Acquire lock before socket operations
                if self.socket_lock.acquire(timeout=1.0):  # 1 second timeout
                    try:
                        # Set a timeout for receiving to avoid blocking forever
                        self.socket.settimeout(0.5)
                        
                        try:
                            data = self.socket.recv(1024).decode()
                            # Reset timeout to blocking mode
                            self.socket.settimeout(None)
                            
                            if data:
                                print("Received from EV3 #" + str(self.ev3_id) + ": " + data)
                            else:
                                # Empty data indicates closed connection
                                print("Connection closed by EV3 #" + str(self.ev3_id))
                                self.connected = False
                        except bluetooth.btcommon.BluetoothError as e:
                            if e.args[0] == 'timed out':
                                # Timeout is normal - just continue
                                pass
                            else:
                                print("Error receiving from EV3 #" + str(self.ev3_id) + ": " + str(e))
                                self.connected = False
                    finally:
                        # Always release the lock
                        self.socket_lock.release()
            
            # Small delay to avoid CPU hogging
            time.sleep(0.1)

    def disconnect(self):
        self.stop_thread = True
        
        # Wait for receive thread to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        
        # Acquire lock before closing socket
        with self.socket_lock:
            if self.socket:
                try:
                    # Proper socket shutdown sequence
                    self.socket.shutdown(bluetooth.SHUT_RDWR)
                except:
                    pass
                finally:
                    try:
                        self.socket.close()
                    except:
                        pass
            
        self.connected = False
        print("Disconnected from EV3 #" + str(self.ev3_id))

def main():
    # Create connections to all EV3s
    ev3_connections = []
    for i, addr in enumerate(ev3_addresses):
        conn = EV3Connection(addr, i+1)
        if conn.connect():
            ev3_connections.append(conn)
    
    try:
        while ev3_connections:
            # Example: send data to all connected EV3s
            message = "CMD" + str(int(time.time()) % 100)
            
            # Check and remove disconnected EV3s
            connected_ev3s = []
            for conn in ev3_connections:
                if conn.connected:
                    success = conn.send_data(message + "_" + str(conn.ev3_id))
                    if success:
                        connected_ev3s.append(conn)
                    else:
                        print("Lost connection to EV3 #" + str(conn.ev3_id))
                        conn.disconnect()  # Clean up the connection
            
            # Update the list with only connected EV3s
            ev3_connections = connected_ev3s
            
            # If all connections are lost, we might want to try reconnecting
            if not ev3_connections:
                print("All connections lost. Trying to reconnect...")
                # Code to attempt reconnection could go here
            
            # Wait before sending again (300ms)
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Clean up connections
        for conn in ev3_connections:
            conn.disconnect()

if __name__ == "__main__":
    main()