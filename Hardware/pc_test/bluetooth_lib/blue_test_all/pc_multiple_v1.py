import bluetooth

# Function to send data to a specific EV3 device
def send_data(ev3_address, message):
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    try:
        sock.connect((ev3_address, 1))  # 1 is the channel number
        sock.send(message)
    except bluetooth.btcommon.BluetoothError as e:
        print(f"Could not connect to {ev3_address}: {e}")
    finally:
        sock.close()

# Function to receive data from an EV3 device
def receive_data(ev3_address):
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.bind(("", 1))
    sock.listen(1)
    print("Waiting for connection...")
    
    client_sock, client_info = sock.accept()
    print(f"Accepted connection from {client_info}")

    try:
        while True:
            data = client_sock.recv(1024)
            if not data:
                break
            print("Received:", data.decode("utf-8"))
    except bluetooth.btcommon.BluetoothError as e:
        print(f"Error: {e}")
    finally:
        client_sock.close()
        sock.close()

# Example usage
ev3_address = "00:16:53:61:CA:75"  # Replace with your EV3 Bluetooth address 00:16:53:61:CA:75 12:16:53:61:CA:75
send_data(ev3_address, "Hello EV3")
receive_data(ev3_address)
