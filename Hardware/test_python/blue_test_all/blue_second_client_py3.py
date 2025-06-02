#!/usr/bin/env python3


import bluetooth
import time

# PC server address and port
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 5
EV3_ID = 2  # Change this for each EV3 (1, 2, or 3)

def connect_to_server():
    try:
        # Create socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        # Connect to PC server
        print("Connecting to PC server...")
        sock.connect((PC_MAC_ADDRESS, PORT))
        
        print("Connected to PC server")
        return sock
    except Exception as e:
        print("Failed to connect:", e)
        return None

def main():
    retry_count = 0
    max_retries = 10
    retry_delay = 5
    
    print("EV3", EV3_ID, "Bluetooth client starting")
    
    try:
        while retry_count < max_retries:
            # Connect to server
            sock = connect_to_server()
            
            if not sock:
                print("Retry", retry_count + 1, "/", max_retries)
                retry_count += 1
                time.sleep(retry_delay)
                continue
            
            # Reset retry count on successful connection
            retry_count = 0
            
            # Message counter
            counter = 0
            
            try:
                # Send initial message
                initial_message = "HELLO_EV3_" + str(EV3_ID)
                sock.send(initial_message.encode())
                print("Sent:", initial_message)
                
                # Main loop: receive from server and send periodic updates
                while True:
                    # Check for messages from server
                    try:
                        sock.settimeout(0.1)  # Short timeout for receive
                        data = sock.recv(1024)
                        
                        if data:
                            message = data.decode()
                            print("Received:", message)
                            
                            # Send acknowledgment
                            ack_message = "ACK_EV3_" + str(EV3_ID) + "_" + message.split("_")[1]
                            sock.send(ack_message.encode())
                            print("Sent ACK:", ack_message)
                    except bluetooth.btcommon.BluetoothError as e:
                        if str(e) != "timed out":
                            raise e
                    
                    # Send periodic data
                    data_message = "DATA_EV3_" + str(EV3_ID) + "_" + str(counter)
                    try:
                        sock.send(data_message.encode())
                        print("Sent data:", data_message)
                    except Exception as e:
                        print("Error sending:", e)
                        break
                    
                    counter += 1
                    time.sleep(0.3)  # 300ms between sends
                
            except Exception as e:
                print("Connection error:", e)
            finally:
                # Close socket
                try:
                    sock.close()
                except:
                    pass
                print("Disconnected from server")
            
            # Wait before reconnecting
            time.sleep(retry_delay)
        
        print("Maximum retries reached. Exiting.")
        
    except KeyboardInterrupt:
        print("Program terminated by user")
        if 'sock' in locals() and sock:
            sock.close()

if __name__ == "__main__":
    main()