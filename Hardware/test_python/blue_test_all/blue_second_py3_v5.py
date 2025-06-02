#!/usr/bin/env python3

import bluetooth
import time

# Configuration
PORT = 1
EV3_ID = 1  # Change this for each EV3

def setup_server():
    server_socket = None
    try:
        # Create socket
        server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        # Bind to port
        server_socket.bind(("", PORT))
        
        # Listen for connections
        server_socket.listen(1)
        
        print("EV3 # waiting for connection on RFCOMM channel ", EV3_ID, PORT)
        return server_socket
    except Exception as e:
        print("Server setup failed: ", e)
        if server_socket:
            try:
                server_socket.close()
            except:
                pass
        return None

def handle_connection(server_socket):
    client_socket = None
    try:
        # Accept connection
        print("Waiting for connection...")
        client_socket, client_info = server_socket.accept()
        print("Accepted connection from ", client_info)
        
        # Receive data
        data = client_socket.recv(1024)
        
        if data:
            # Process data
            message = data.decode()
            print("Received: ", message)
            
            # Send response
            response = "RESPONSE_EV3_"+str(EV3_ID)+"_"+(message.split('_')[-1] if '_' in message else 'UNKNOWN')
            client_socket.send(response.encode())
            print("Sent: ", response)
        
        # Close client socket
        client_socket.close()
        return True
    except Exception as e:
        print("Error handling connection: ", e)
        if client_socket:
            try:
                client_socket.close()
            except:
                pass
        return False

def main():
    print("Starting EV3 # Bluetooth server", EV3_ID)
    
    try:
        # Set up server
        server_socket = setup_server()
        if not server_socket:
            print("Failed to set up server. Exiting.")
            return
        
        # Main loop - handle connections one by one
        while True:
            handle_connection(server_socket)
            # Small delay between connections
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Clean up
        if 'server_socket' in locals() and server_socket:
            server_socket.close()
            print("Server socket closed")

if __name__ == "__main__":
    main()