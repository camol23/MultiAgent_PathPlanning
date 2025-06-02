#!/usr/bin/env python3

import bluetooth
import time
import random

# PC server address and port
PC_MAC_ADDRESS = "40:23:43:76:A9:D2"  # Replace with your PC's Bluetooth MAC address
PORT = 5
EV3_ID = 1  # Change this for each EV3: 1, 2, or 3

def connect_to_server():
    sock = None
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
        if sock:
            try:
                sock.close()
            except:
                pass
        return None

def main():
    retry_count = 0
    max_retries = 10
    
    # Stagger connection attempts based on EV3_ID
    # First EV3 tries immediately, others wait
    startup_delay = (EV3_ID - 1) * 5
    if startup_delay > 0:
        print("EV3", EV3_ID, "waiting", startup_delay, "seconds before starting...")
        time.sleep(startup_delay)
    
    print("EV3", EV3_ID, "Bluetooth client starting")
    
    while retry_count < max_retries:
        # Try to clean up any old sockets
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
            # Add some randomness to retry delay based on EV3_ID
            retry_delay = 5 + (EV3_ID * 2) + random.randint(1, 5)
            time.sleep(retry_delay)
            continue
        
        # Reset retry count on successful connection
        retry_count = 0
        counter = 0
        
        try:
            # Send initial hello message with EV3_ID
            initial_message = "HELLO_EV3_" + str(EV3_ID)
            sock.send(initial_message.encode())
            print("Sent:", initial_message)
            
            # Wait for welcome message
            try:
                data = sock.recv(1024)
                if data:
                    message = data.decode()
                    print("Received:", message)
            except Exception as e:
                print("Error receiving welcome:", e)
            
            # Main communication loop
            last_send_time = time.time()
            
            while True:
                # Try to receive messages
                try:
                    # Non-blocking receive attempt
                    sock.setblocking(0)
                    try:
                        data = sock.recv(1024)
                        if data:
                            message = data.decode()
                            print("Received:", message)
                            
                            # Send acknowledgment for CMD messages
                            if message.startswith("CMD_"):
                                parts = message.split("_")
                                if len(parts) > 1:
                                    ack_message = "ACK_EV3_" + str(EV3_ID) + "_" + parts[1]
                                    sock.setblocking(1)  # Switch to blocking for send
                                    sock.send(ack_message.encode())
                                    print("Sent ACK:", ack_message)
                    except bluetooth.btcommon.BluetoothError:
                        # Expected for non-blocking socket with no data
                        pass
                except Exception as e:
                    print("Error receiving:", e)
                    break
                
                # Send periodic data
                current_time = time.time()
                if current_time - last_send_time >= 0.3:  # 300ms between sends
                    data_message = "DATA_EV3_" + str(EV3_ID) + "_" + str(counter)
                    try:
                        sock.setblocking(1)  # Switch to blocking for send
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
        
        # Wait before trying to reconnect
        retry_delay = 5 + (EV3_ID * 2)
        print("Waiting", retry_delay, "seconds before reconnecting...")
        time.sleep(retry_delay)
    
    print("Maximum retries reached. Exiting.")

if __name__ == "__main__":
    main()