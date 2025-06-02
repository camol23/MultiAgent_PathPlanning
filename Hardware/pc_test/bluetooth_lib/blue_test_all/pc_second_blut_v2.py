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

ev3_addresses = [
    "F0:45:DA:11:92:74",  # EV3 #
    "F0:45:DA:D2:C8:88"
]

# RFCOMM port (must match on EV3s)
port = 5

class EV3Connection:
    def __init__(self, mac_address, ev3_id):
        self.mac_address = mac_address
        self.ev3_id = ev3_id
        self.socket = None
        self.connected = False
        self.stop_thread = False
        self.receive_thread = None


    def connect(self):
        try:
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.connect((self.mac_address, port))
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
        if self.connected:
            try:
                self.socket.send(data.encode())

                return True
            except Exception as e:
                print("Error sending to EV3 #" + str(self.ev3_id) + ": " + str(e))
                self.connected = False
                return False
        return False

    def receive_data(self):
        while not self.stop_thread:
            if self.connected:
                try:
                    data = self.socket.recv(1024).decode()
                    if data:
                        print("Received from EV3 #" + str(self.ev3_id) + ": " + data)
                    else:
                        # Empty data might indicate connection closed
                        print("Connection to EV3 #" + str(self.ev3_id) + " might be broken, empty data received")
                        time.sleep(1)  # Wait before attempting reconnection
                        self.reconnect()
                        break
                except bluetooth.btcommon.BluetoothError as e:
                    print("Error receiving from EV3 #" + str(self.ev3_id) + ": " + str(e))
                    self.connected = False
                    time.sleep(1)  # Wait before attempting reconnection
                    self.reconnect()
                    break
            time.sleep(0.1)  # 100ms pause to avoid high CPU usage
    
    def reconnect(self):
        # Close existing socket if any
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        # Try to reconnect
        print("Attempting to reconnect to EV3 #" + str(self.ev3_id))
        for attempt in range(3):  # Try 3 times
            try:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.mac_address, port))
                self.connected = True
                print("Reconnected to EV3 #" + str(self.ev3_id))
                return True
            except Exception as e:
                print(f"Reconnection attempt {attempt+1} failed: {e}")
                time.sleep(2)  # Wait between attempts
        
        print("Failed to reconnect to EV3 #" + str(self.ev3_id))
        return False

    def disconnect(self):
        self.stop_thread = True
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
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
            
            for i, conn in enumerate(ev3_connections):
                success = conn.send_data(message + "_" + str(i+1))
                if not success:
                    print("Lost connection to EV3 #" + str(conn.ev3_id))
            
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