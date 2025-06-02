#!/usr/bin/env python3

import bluetooth

def start_bluetooth_server():
    # Create a Bluetooth socket
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

    # Bind the socket to an address and port
    server_socket.bind(("", bluetooth.PORT_ANY))

    # Start listening for incoming connections
    server_socket.listen(1)

    # Get the port the server is using
    port = server_socket.getsockname()[1]

    # Advertise the service
    bluetooth.advertise_service(
        server_socket,
        "EV3BluetoothService",
        service_id="00001101-0000-1000-8000-00805F9B34FB",
        service_classes=["00001101-0000-1000-8000-00805F9B34FB", bluetooth.SERIAL_PORT_PROFILE],
    )

    # print(f"Waiting for connections on RFCOMM channel {port}...")
    print("Waiting for connections on RFCOMM channel ", port)

    while True:
        # Accept a connection
        client_socket, client_info = server_socket.accept()
        # print(f"Accepted connection from {client_info}")
        print("Accepted connection from ",client_info)

        try:
            while True:
                data = client_socket.recv(1024)  # Receive data from the client
                if not data:
                    break
                
                # Decode the received bytes to string
                message = data.decode('utf-8').strip()
                # print(f"Received message: {message}")
                print("Received message: ", message)
                
                # Process the message if needed
                # Example: You can split and handle temperature and light values here

        except OSError:
            pass
        finally:
            client_socket.close()
            print("Connection closed")

if __name__ == "__main__":
    start_bluetooth_server()