#!/usr/bin/env python3
import subprocess
import time
import threading
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import sys
import os

# BLE Service UUIDs and characteristics
BLUEZ_SERVICE_NAME = "org.bluez"
ADAPTER_INTERFACE = "org.bluez.Adapter1"
DEVICE_INTERFACE = "org.bluez.Device1"
GATT_MANAGER_INTERFACE = "org.bluez.GattManager1"
LE_ADVERTISING_MANAGER_INTERFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_INTERFACE = "org.bluez.LEAdvertisement1"


class Advertisement(dbus.service.Object):
    def __init__(self, bus, path, ad_props):
        self.path = path
        self.props = ad_props
        dbus.service.Object.__init__(self, bus, path)

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                f"Interface {interface} not supported")
        return self.props

    @dbus.service.method(dbus.PROPERTIES_IFACE, in_signature="ss", out_signature="v")
    def Get(self, interface, prop):
        if interface != LE_ADVERTISEMENT_INTERFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                f"Interface {interface} not supported")
        return self.props.get(prop, None)

    @dbus.service.method(LE_ADVERTISEMENT_INTERFACE)
    def Release(self):
        print("Advertisement released")


class BLEAdvertisement:
    def __init__(self):
        self.bus = None
        self.adapter = None
        self.mainloop = None
        self.ad_manager = None
        self.advertisement = None
        self.running = False
        # Make path unique with timestamp
        self.path = f"/org/bluez/pc_adv_{int(time.time())}"
        
    def setup(self):
        # Initialize D-Bus
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()
        self.mainloop = GLib.MainLoop()
        
        # Get the adapter
        self.adapter = self.find_adapter()
        if not self.adapter:
            print("Bluetooth adapter not found")
            return False
        
        # Get the LE advertising manager
        self.ad_manager = self.get_ad_manager()
        if not self.ad_manager:
            print("LEAdvertisingManager1 interface not found")
            return False
            
        print("BLE Advertisement setup complete")
        return True
    
    def find_adapter(self):
        remote_om = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, "/"),
                                   "org.freedesktop.DBus.ObjectManager")
        objects = remote_om.GetManagedObjects()
        
        for obj_path, interfaces in objects.items():
            if ADAPTER_INTERFACE in interfaces:
                return obj_path
                
        return None
    
    def get_ad_manager(self):
        return dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, self.adapter),
                             LE_ADVERTISING_MANAGER_INTERFACE)
    
    def create_advertisement(self, message, target_id=0):
        # Limit message length to ensure it fits in advertisement (max ~20 chars)
        if len(message) > 20:
            message = message[:20]
            
        # Create a dict for the advertisement
        ad_props = {
            "Type": dbus.String("peripheral"),
            "ServiceUUIDs": dbus.Array(["180d"], signature="s"),  # Heart Rate Service UUID
            "LocalName": dbus.String("PC-Master"),
            "ManufacturerData": dbus.Dictionary({
                dbus.UInt16(0xFFFF): dbus.Array([
                    dbus.Byte(target_id),  # Target board ID (0 = broadcast)
                    *[dbus.Byte(c) for c in message.encode()]  # Actual message
                ], signature=dbus.Signature("y"))
            }, signature=dbus.Signature("qv"))
        }
        
        # Unregister any existing advertisement
        if self.advertisement:
            try:
                self.ad_manager.UnregisterAdvertisement(self.advertisement.path)
                self.advertisement.remove_from_connection()
            except Exception as e:
                print(f"Error unregistering previous advertisement: {e}")
        
        # Create and register the new advertisement
        try:
            self.advertisement = Advertisement(self.bus, self.path, ad_props)
            
            self.ad_manager.RegisterAdvertisement(
                self.path,
                dbus.Dictionary({}, signature="sv"),
                reply_handler=self.register_ad_cb,
                error_handler=self.register_ad_error_cb
            )
            return True
        except Exception as e:
            print(f"Failed to create advertisement: {e}")
            return False
    
    def register_ad_cb(self):
        print("Advertisement registered")
    
    def register_ad_error_cb(self, error):
        print(f"Failed to register advertisement: {error}")
        self.mainloop.quit()
    
    def start_advertising(self, message, target_id=0):
        if not self.create_advertisement(message, target_id):
            return False
        
        # Start the main loop if not already running
        if not self.running:
            self.running = True
            self.mainloop_thread = threading.Thread(target=self.run_mainloop)
            self.mainloop_thread.daemon = True
            self.mainloop_thread.start()
        
        return True
    
    def run_mainloop(self):
        self.mainloop.run()
    
    def stop_advertising(self):
        if self.advertisement:
            try:
                self.ad_manager.UnregisterAdvertisement(self.path)
                self.advertisement.remove_from_connection()
                self.advertisement = None
            except Exception as e:
                print(f"Error unregistering advertisement: {e}")
        
        self.running = False
        if self.mainloop and self.mainloop.is_running():
            self.mainloop.quit()


class BLEScanner:
    def __init__(self):
        self.running = False
        self.boards_data = {1: {}, 2: {}, 3: {}}  # Store data from each board
        self.scan_process = None
        self.dump_process = None
        
    def start_scanning(self):
        if self.running:
            return
            
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
        # Start the packet monitor in separate thread
        self.monitor_thread = threading.Thread(target=self._monitor_advertisements)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("BLE scanner started")
    
    def _scan_loop(self):
        try:
            while self.running:
                # Use hcitool to scan for advertisements
                self.scan_process = subprocess.Popen(
                    ["hcitool", "lescan", "--duplicates"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait before restarting scan (hcitool sometimes stalls)
                time.sleep(30)
                
                # Kill the process
                if self.scan_process:
                    self.scan_process.terminate()
                    self.scan_process = None
                    time.sleep(1)  # Small delay before restarting
                
        except Exception as e:
            print(f"Scan error: {e}")
        finally:
            self.running = False
    
    def _monitor_advertisements(self):
        try:
            # Use hcidump to monitor raw advertisement data
            self.dump_process = subprocess.Popen(
                ["hcidump", "--raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            buffer = b""
            while self.running:
                # Read data from hcidump
                data = self.dump_process.stdout.read(1)
                if not data:
                    break
                    
                buffer += data
                
                # Check for complete packets
                if b"\n" in buffer:
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()  # Keep the incomplete line
                    
                    # Process complete lines
                    for line in lines:
                        self._process_adv_data(line.decode('utf-8', errors='replace').strip())
            
        except Exception as e:
            print(f"Error monitoring advertisements: {e}")
        finally:
            # Clean up
            if self.dump_process:
                try:
                    self.dump_process.terminate()
                except:
                    pass
                self.dump_process = None
    
    def _process_adv_data(self, line):
        # Looking for EV3 board manufacturer data (with 0xFFFF identifier)
        if "EV3-" in line and "ff ff" in line.lower():
            try:
                # Try to identify which board
                if "EV3-1" in line:
                    board_id = 1
                elif "EV3-2" in line:
                    board_id = 2
                elif "EV3-3" in line:
                    board_id = 3
                else:
                    return  # Not a board we're tracking
                
                # Extract raw hex data following 0xFF 0xFF (manufacturer specific data)
                parts = line.lower().split("ff ff")
                if len(parts) < 2:
                    return
                
                # The first byte after FF FF should be the board ID, followed by the message in ASCII
                hex_data = parts[1].strip().replace(" ", "")
                
                # Convert hex to ASCII (skipping the first byte which is the board ID)
                try:
                    # Skip first byte (board ID) and convert remaining bytes to ASCII
                    ascii_message = bytes.fromhex(hex_data[2:]).decode('ascii', errors='replace')
                    
                    # Update board data
                    self.boards_data[board_id]['last_message'] = ascii_message
                    self.boards_data[board_id]['last_seen'] = time.time()
                    
                    print(f"Received from Board {board_id}: {ascii_message}")
                except Exception as e:
                    print(f"Error parsing data from board {board_id}: {e}")
                    
            except Exception as e:
                print(f"Error processing advertisement: {e}")
    
    def stop_scanning(self):
        self.running = False
        if self.scan_process:
            self.scan_process.terminate()
            self.scan_process = None
        if self.dump_process:
            self.dump_process.terminate()
            self.dump_process = None


def broadcast_message(advertiser, message, target_id=0):
    print(f"Broadcasting to Board {target_id if target_id else 'ALL'}: {message}")
    advertiser.start_advertising(message, target_id)
    # Brief pause to ensure message is broadcast
    time.sleep(0.5)


def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("This script must be run as root. Try using sudo.")
        sys.exit(1)
    
    # Initialize BLE advertisement
    advertiser = BLEAdvertisement()
    if not advertiser.setup():
        print("Failed to set up BLE advertisement")
        sys.exit(1)
    
    # Initialize scanner
    scanner = BLEScanner()
    scanner.start_scanning()
    
    message_counter = 0
    try:
        while True:
            # Choose which board to send to
            if message_counter % 4 == 0:
                # Broadcast to all boards every 4th message
                message = f"ALL:{message_counter}"
                broadcast_message(advertiser, message, target_id=0)
            else:
                # Round robin to each board
                target_board = (message_counter % 3) + 1  # Boards 1, 2, 3
                message = f"B{target_board}:{message_counter}"
                broadcast_message(advertiser, message, target_id=target_board)
            
            # Display latest data from all boards
            print("\nLatest data from boards:")
            for board_id, data in scanner.boards_data.items():
                if data.get('last_seen'):
                    time_ago = time.time() - data['last_seen']
                    print(f"Board {board_id}: {data.get('last_message', 'No data')} "
                          f"({time_ago:.1f}s ago)")
                else:
                    print(f"Board {board_id}: No data received yet")
            print()
            
            # Wait between broadcasts
            time.sleep(2)
            message_counter += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        advertiser.stop_advertising()
        scanner.stop_scanning()
        print("PC broadcaster stopped")


if __name__ == "__main__":
    main()