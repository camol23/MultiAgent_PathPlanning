#!/usr/bin/env python3

# from ev3dev2.motor import LargeMotor, OUTPUT_A
import bluetooth
import threading
import time

class BluetoothServer:
    def __init__(self):
        self.server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.server_sock.bind(("", 1))
        self.server_sock.listen(1)

    def handle_client(self, client_sock):
        try:
            while True:
                data = client_sock.recv(1024)
                if not data:
                    break
                print("Received:", data.decode("utf-8"))
                client_sock.send("Message received")
        except bluetooth.btcommon.BluetoothError as e:
            # print(f"Bluetooth error: {e}")
            print("Bluetooth error: ", e)
        finally:
            client_sock.close()

    def start(self):
        print("Waiting for connection...")
        client_sock, client_info = self.server_sock.accept()
        print("Accepted connection from", client_info)
        
        threading.Thread(target=self.handle_client, args=(client_sock,)).start()
        
        # Continuously send data
        while True:
            client_sock.send("Data from EV3")
            time.sleep(1)  # Send data every second

if __name__ == "__main__":
    server = BluetoothServer()
    server.start()
