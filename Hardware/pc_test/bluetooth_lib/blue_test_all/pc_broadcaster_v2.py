import subprocess
import time
import binascii
import threading
import bluetooth
import struct
from bluetooth.ble import DiscoveryService, GATTRequester

def create_adv_packet(message, target_id=0):
    # Convert message to hex
    msg_hex = binascii.hexlify(message.encode()).decode()
    
    # Create advertising data
    # Set flags (LE General Discoverable + BR/EDR Not Supported)
    flags = "0201060"
    
    # Add complete local name
    name = "PC-Master"
    name_hex = binascii.hexlify(name.encode()).decode()
    name_type = "09"  # Complete Local Name
    name_len = format(len(name) + 1, 'x').zfill(2)
    name_data = name_len + name_type + name_hex
    
    # Add manufacturer specific data with target ID and message
    manuf_id = "FFFF"  # For testing/development
    manuf_type = "FF"  # Manufacturer Specific Data
    
    # Include target ID (0 = broadcast to all, 1-3 = specific board)
    target_hex = format(target_id, 'x').zfill(2)
    manuf_data = target_hex + msg_hex
    
    manuf_len = format(len(manuf_data)//2 + 3, 'x').zfill(2)  # +3 for type and manuf_id
    manuf_full = manuf_len + manuf_type + manuf_id + manuf_data
    
    # Combine all parts
    adv_data = flags + name_data + manuf_full
    return adv_data

def broadcast_message(message, target_id=0):
    adv_data = create_adv_packet(message, target_id)
    
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
        
        # Set advertising parameters
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x0006",
                       "A0", "00", "A0", "00", "03", "00", "00", "00",
                       "00", "00", "00", "00", "00", "07", "00"], check=True)
        
        # Enable advertising
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x000A", "01"], check=True)
        
        print(f"Broadcasting to Board {target_id if target_id else 'ALL'}: {message}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

# Scanner class to receive messages from MicroPython boards
class BLEScanner:
    def __init__(self):
        self.running = False
        self.boards_data = {1: {}, 2: {}, 3: {}}  # Store data from each board
    
    def process_advertisement(self, addr, rssi, adv_data):
        try:
            # Parse the advertisement data
            i = 0
            board_id = None
            message = None
            
            while i < len(adv_data):
                if i + 1 >= len(adv_data):
                    break
                    
                length = adv_data[i]
                if i + length + 1 > len(adv_data):
                    break
                
                adv_type = adv_data[i+1]
                
                # Check for manufacturer specific data
                if adv_type == 0xFF and length >= 5:  # Type + ManufID(2) + BoardID(1) + Data(1+)
                    # Check our manufacturer ID (FFFF)
                    if adv_data[i+2:i+4] == b'\xFF\xFF':
                        # Extract board ID from first byte of data
                        board_id = adv_data[i+4]
                        
                        # Extract message from remaining data
                        message_bytes = adv_data[i+5:i+length+1]
                        try:
                            message = message_bytes.decode('utf-8')
                        except:
                            message = binascii.hexlify(message_bytes).decode('utf-8')
                
                i += length + 1
            
            if board_id is not None and 1 <= board_id <= 3:
                self.boards_data[board_id]['last_message'] = message
                self.boards_data[board_id]['last_rssi'] = rssi
                self.boards_data[board_id]['last_seen'] = time.time()
                print(f"Received from Board {board_id}: {message}, RSSI: {rssi}dB")
                
        except Exception as e:
            print(f"Error processing advertisement: {e}")
    
    # Start scanning in a separate thread
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        print("BLE scanner started")
    
    def _scan_loop(self):
        try:
            # Set up scanning with PyBluez and Linux hcitool
            while self.running:
                # Use hcitool to scan (simplified, real implementation would use callbacks)
                process = subprocess.Popen(
                    ["sudo", "hcitool", "lescan", "--duplicates"],
                    stdout=subprocess.PIPE
                )
                
                # Wait briefly to collect some advertisements
                time.sleep(5)
                process.terminate()
                
                # Process results (in real implementation, use proper BLE libraries like bleak)
                # This is just a placeholder - actual implementation needs a proper BLE library
                
                # Simulate receiving data - in a real implementation, parse real BLE data
                self._simulate_received_data()
                
                # Brief pause before next scan
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Scan error: {e}")
        finally:
            self.running = False
    
    # Temporary method to simulate receiving data (remove in real implementation)
    def _simulate_received_data(self):
        # This is just for illustration - replace with actual BLE parsing
        pass
    
    def stop_scanning(self):
        self.running = False
        if hasattr(self, 'scan_thread'):
            self.scan_thread.join(timeout=1.0)
        print("BLE scanner stopped")

def main():
    scanner = BLEScanner()
    scanner.start_scanning()
    
    message_counter = 0
    try:
        while True:
            # Example of sending to all boards
            if message_counter % 5 == 0:
                message = f"ALL:{message_counter}"
                broadcast_message(message, target_id=0)  # 0 = broadcast to all
            
            # Example of sending to individual boards
            else:
                target_board = (message_counter % 3) + 1  # Boards 1, 2, 3
                message = f"MSG:{message_counter}"
                broadcast_message(message, target_id=target_board)
            
            # Display latest data from all boards
            print("\nLatest data from boards:")
            for board_id, data in scanner.boards_data.items():
                if data.get('last_seen'):
                    time_ago = time.time() - data['last_seen']
                    print(f"Board {board_id}: {data.get('last_message', 'No data')} "
                          f"(RSSI: {data.get('last_rssi')}dB, {time_ago:.1f}s ago)")
                else:
                    print(f"Board {board_id}: No data received yet")
            print()
            
            time.sleep(2)
            message_counter += 1
            
    except KeyboardInterrupt:
        # Disable advertising before exit
        subprocess.run(["sudo", "hcitool", "cmd", "0x08", "0x000A", "00"], check=True)
        scanner.stop_scanning()
        print("Broadcasting and scanning stopped")

if __name__ == "__main__":
    main()