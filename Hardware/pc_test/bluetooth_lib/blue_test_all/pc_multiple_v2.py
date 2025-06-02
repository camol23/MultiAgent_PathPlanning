import bluetooth
import threading
import time

class BluetoothClient:
    def __init__(self, ev3_address):
        self.ev3_address = ev3_address
        self.sock = None

    def connect(self):
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            self.sock.connect((self.ev3_address, 1))
            print(f"Connected to {self.ev3_address}")
        except bluetooth.btcommon.BluetoothError as e:
            print(f"Could not connect to {self.ev3_address}: {e}")

    def send_data(self, message):
        while True:
            if self.sock:
                try:
                    self.sock.send(message)
                    time.sleep(1)  # Send message every second
                except bluetooth.btcommon.BluetoothError as e:
                    print(f"Error sending data to {self.ev3_address}: {e}")
                    break

    def receive_data(self):
        if self.sock:
            try:
                while True:
                    data = self.sock.recv(1024)
                    if not data:
                        break
                    print(f"Received from {self.ev3_address}: {data.decode('utf-8')}")
            except bluetooth.btcommon.BluetoothError as e:
                print(f"Error receiving data from {self.ev3_address}: {e}")

    def close(self):
        if self.sock:
            self.sock.close()

def main():
    # ev3_addresses = ["00:00:00:00:00:00", "00:00:00:00:00:01", "00:00:00:00:00:02"]  # Replace with actual EV3 addresses
    ev3_addresses = ["00:16:53:61:CA:75"]
    clients = [BluetoothClient(addr) for addr in ev3_addresses]

    for client in clients:
        client.connect()  # Connect to the EV3 devices

    threads = []
    for client in clients:
        t_send = threading.Thread(target=client.send_data, args=("Hello EV3",))
        t_receive = threading.Thread(target=client.receive_data)
        threads.append(t_send)
        threads.append(t_receive)
        t_send.start()
        t_receive.start()

    for thread in threads:
        thread.join()  # Wait for threads to finish

    for client in clients:
        client.close()  # Clean up sockets

if __name__ == "__main__":
    main()
