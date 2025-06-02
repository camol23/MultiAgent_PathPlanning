#!/usr/bin/env python3

#from ev3dev2.motor import LargeMotor, OUTPUT_A
import bluetooth

def setup_bluetooth():
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", 1))
    server_sock.listen(1)

    print("Waiting for connection...")
    client_sock, client_info = server_sock.accept()
    print("Accepted connection from", client_info)

    return client_sock

def main():
    client_sock = setup_bluetooth()

    try:
        while True:
            data = client_sock.recv(1024)
            if not data:
                break
            print("Received:", data.decode("utf-8"))
            # Send a response
            client_sock.send("Message received")

    except bluetooth.btcommon.BluetoothError as e:
        # print(f"Bluetooth error: {e}")
        print("Bluetooth error: ", e)
    finally:
        client_sock.close()

if __name__ == "__main__":
    main()
