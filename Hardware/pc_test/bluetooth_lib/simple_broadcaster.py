import subprocess
import time
import binascii

def create_adv_packet(message):
    # Convert message to hex
    msg_hex = binascii.hexlify(message.encode()).decode()
    
    # Create advertising data
    # Set flags (LE General Discoverable + BR/EDR Not Supported)
    flags = "0201060"
    
    # Add complete local name
    name = "PC-Broadcaster"
    name_hex = binascii.hexlify(name.encode()).decode()
    name_type = "09"  # Complete Local Name
    name_len = format(len(name) + 1, 'x').zfill(2)
    name_data = name_len + name_type + name_hex
    
    # Add manufacturer specific data
    manuf_id = "FFFF"  # For testing/development
    manuf_type = "FF"  # Manufacturer Specific Data
    manuf_data = msg_hex
    manuf_len = format(len(manuf_data)//2 + 3, 'x').zfill(2)  # +3 for type and manuf_id
    manuf_full = manuf_len + manuf_type + manuf_id + manuf_data
    
    # Combine all parts
    adv_data = flags + name_data + manuf_full
    
    return adv_data

def broadcast_message(message):
    adv_data = create_adv_packet(message)
    
    # Setup LE advertising using hciconfig and hcitool
    try:
        # Reset adapter
        subprocess.run(["sudo", "hciconfig", "hci0", "down"], check=True)
        subprocess.run(["sudo", "hciconfig", "hci0", "up"], check=True)
        
        # Stop any existing advertisements
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x000A", "00"], check=True)
        
        # Set advertising data
        cmd = ["sudo", "hcitool", "cmd", "0x08", "0x0008"]
        for i in range(0, len(adv_data), 2):
            if i < len(adv_data):
                cmd.append(adv_data[i:i+2])
        
        subprocess.run(cmd, check=True)
        
        # Set advertising parameters (interval, etc.)
        # 0x00A0 = 160ms, 0x00A0 = 160ms (min and max interval)
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x0006", 
                        "A0", "00", "A0", "00", "03", "00", "00", "00", 
                        "00", "00", "00", "00", "00", "07", "00"], check=True)
        
        # Enable advertising
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x000A", "01"], check=True)
        
        print(f"Broadcasting: {message}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def main():
    message_counter = 0
    
    try:
        while True:
            # message = f"Broadcast #{message_counter} @ {time.time():.2f}"
            message = f"Broadcast #{message_counter}"
            broadcast_message(message)
            time.sleep(1)
            message_counter += 1
            
    except KeyboardInterrupt:
        # Disable advertising before exit
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x000A", "00"], check=True)
        print("Broadcasting stopped")

if __name__ == "__main__":
    main()