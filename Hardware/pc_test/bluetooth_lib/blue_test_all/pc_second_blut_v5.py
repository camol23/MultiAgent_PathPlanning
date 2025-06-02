import bluetooth
import time

# EV3 device MAC address
ev3_address = "F0:45:DA:11:92:74"  # Replace with your actual EV3 MAC address
port = 1

def send_message_to_ev3(message):
    try:
        # Create new socket for each transmission
        socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        print(f"Connecting to EV3 at {ev3_address}...")
        socket.connect((ev3_address, port))
        print("Connected")
        
        # Allow connection to stabilize
        time.sleep(0.5)
        
        # Send message
        print(f"Sending: {message}")
        socket.send(message.encode())
        
        # Wait for response with timeout
        socket.settimeout(2.0)
        try:
            response = socket.recv(1024).decode()
            print(f"Received: {response}")
        except bluetooth.btcommon.BluetoothError:
            print("No response received")
            
        # Close socket
        socket.close()
        print("Disconnected")
        return True
    except Exception as e:
        print(f"Error: {e}")
        try:
            socket.close()
        except:
            pass
        return False

def main():
    counter = 0
    try:
        while True:
            # Create unique message
            message = f"CMD_{counter}"
            
            # Send message to EV3
            success = send_message_to_ev3(message)
            
            # Increment counter
            counter += 1
            
            # Wait between transmissions
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("Program terminated by user")

if __name__ == "__main__":
    main()