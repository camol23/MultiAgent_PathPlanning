#!/usr/bin/env python3

import bluetooth
import time
import random
import sys

# PC server address and port
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 5
EV3_ID = 1  # Change this for each EV3 (1, 2, or 3)

def connect_to_server():
    sock = None
    try:
        # Create socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        
        # Connect to PC server with timeout
        print("Connecting to PC server...")
        sock.connect((PC_MAC_ADDRESS, PORT))
        
        print("Connected to PC server")
        return sock
    except Exception as e:
        print("Failed to connect:", e)
        if sock:
            try:
                sock.close()
            except:
                pass
        return None

def main():
    retry_count = 0
    max_retries = 10
    retry_delay = 5
    
    # Add small random delay based on EV3_ID to stagger connections
    startup_delay = (EV3_ID - 1) * 3.0
    # print(f"EV3 {EV3_ID} waiting {startup_delay:.1f} seconds before starting...")
    print("EV3 __ waiting __ seconds before starting...", EV3_ID, startup_delay)
    time.sleep(startup_delay)
    
    print("EV3", EV3_ID, "Bluetooth client starting")
    
    try:
        while retry_count < max_retries:
            # Explicitly close any lingering sockets before creating new one
            # This might help with "File descriptor in bad state" errors
            for i in range(3):  # Force garbage collection
                try:
                    dummy_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                    dummy_sock.close()
                except:
                    pass
            
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
                
                # Wait for welcome message
                try:
                    data = sock.recv(1024)
                    if data:
                        welcome = data.decode()
                        print("Received welcome:", welcome)
                except Exception as e:
                    print("No welcome received:", e)
                
                # Main loop: receive from server and send periodic updates
                last_send_time = time.time()
                
                while True:
                    # Check for messages from server
                    try:
                        data = None
                        # Non-blocking receive
                        sock.setblocking(0)
                        try:
                            data = sock.recv(1024)
                        except bluetooth.btcommon.BluetoothError as e:
                            if "resource temporarily unavailable" not in str(e).lower():
                                raise e
                        
                        if data:
                            message = data.decode()
                            print("Received:", message)
                            
                            # Send acknowledgment
                            if message.startswith("CMD_"):
                                parts = message.split("_")
                                if len(parts) > 1:
                                    cmd_num = parts[1]
                                    ack_message = "ACK_EV3_" + str(EV3_ID) + "_" + cmd_num
                                    sock.send(ack_message.encode())
                                    print("Sent ACK:", ack_message)
                    except Exception as e:
                        if "timed out" not in str(e).lower():
                            print("Error receiving:", e)
                            break
                    
                    # Send periodic data with fixed interval
                    current_time = time.time()
                    if current_time - last_send_time >= 0.3:  # 300ms
                        data_message = "DATA_EV3_" + str(EV3_ID) + "_" + str(counter)
                        try:
                            sock.setblocking(1)  # Switch back to blocking for send
                            sock.send(data_message.encode())
                            print("Sent data:", data_message)
                            counter += 1
                            last_send_time = current_time
                        except Exception as e:
                            print("Error sending:", e)
                            break
                    
                    # Small wait to prevent CPU hogging
                    time.sleep(0.05)
                
            except Exception as e:
                print("Connection error:", e)
            finally:
                # Close socket
                try:
                    sock.close()
                except:
                    pass
                print("Disconnected from server")
            
            # Add a slightly randomized retry delay
            actual_delay = retry_delay + random.uniform(0.1, 1.0) + (EV3_ID * 1.0)
            # print(f"Waiting {actual_delay:.1f} seconds before reconnecting...")
            print("Waiting __ seconds before reconnecting...", actual_delay)
            time.sleep(actual_delay)
        
        print("Maximum retries reached. Exiting.")
        
    except KeyboardInterrupt:
        print("Program terminated by user")
        if 'sock' in locals() and sock:
            try:
                sock.close()
            except:
                pass

if __name__ == "__main__":
    main()