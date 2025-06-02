#!/usr/bin/env python3
import subprocess
import time
import binascii
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
        # Make path unique with timestamp or random ID
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
        
        # # Add methods to the advertisement object
        # def get_properties():
        #     return ad_props
        
        # def get_path():
        #     return self.path
            
        # def release():
        #     print("Advertisement released")
            
        # self.advertisement.Get = lambda interface, prop: ad_props.get(prop)
        # self.advertisement.GetAll = lambda interface: ad_props
        # self.advertisement.GetPath = get_path
        # self.advertisement.Release = release
        
        # self.advertisement._dbus_interface = LE_ADVERTISEMENT_INTERFACE
        
        # # Register the advertisement
        # self.ad_manager.RegisterAdvertisement(
        #     self.advertisement.GetPath(),
        #     {},
        #     reply_handler=self.register_ad_cb,
        #     error_handler=self.register_ad_error_cb
        # )
        
        # return True
    
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
            while self.running:
                # Use hcitool to scan for advertisements
                self.scan_process = subprocess.Popen(
                    ["hcitool", "lescan", "--duplicates"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Start a separate thread to read from the process
                read_thread = threading.Thread(target=self._read_scan_output)
                read_thread.daemon = True
                read_thread.start()
                
                # Also monitor for advertisement data
                monitor_thread = threading.Thread(target=self._monitor_advertisements)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Wait before restarting scan
                time.sleep(10)
                
                # Kill the process
                if self.scan_process:
                    self.scan_process.terminate()
                    self.scan_process = None
                
        except Exception as e:
            print(f"Scan error: {e}")
        finally:
            self.running = False
    
    def _read_scan_output(self):
        try:
            while self.scan_process and self.running:
                line = self.scan_process.stdout.readline().decode('utf-8').strip()
                if line:
                    # Process the scan output
                    self._process_scan_line(line)
        except Exception as e:
            print(f"Error reading scan output: {e}")
    
    def _process_scan_line(self, line):
        parts = line.split(' ')
        if len(parts) >= 2:
            mac_address = parts[0]
            name = ' '.join(parts[1:])
            
            # Check if this is from one of our EV3 boards
            if "EV3-" in name:
                board_id = None
                try:
                    # Extract board ID from name (e.g., "EV3-1")
                    board_id = int(name.split('-')[1])
                except:
                    pass
                
                if board_id and 1 <= board_id <= 3:
                    # Update the board's MAC address
                    self.boards_data[board_id]['mac'] = mac_address
                    self.boards_data[board_id]['name'] = name
    
    def _monitor_advertisements(self):
        try:
            # Use hcidump to monitor raw advertisement data
            proc = subprocess.Popen(
                ["hcidump", "--raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            while self.running:
                # line = proc.stdout.readline().decode('utf-8').strip()
                line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
                if line:
                    # Here you would parse the raw bluetooth packets
                    # This is a simplified placeholder - real implementation would need
                    # proper BLE packet parsing
                    if "FFFF" in line and ("EV3-1" in line or "EV3-2" in line or "EV3-3" in line):
                        # Try to extract board ID and message
                        # This is highly simplified - would need proper parsing
                        try:
                            if "EV3-1" in line:
                                board_id = 1
                            elif "EV3-2" in line:
                                board_id = 2
                            elif "EV3-3" in line:
                                board_id = 3
                            else:
                                continue
                                
                            # Extract message - this is a placeholder
                            # In reality, you'd parse the actual BLE packet
                            message = line.split("FFFF")[1].strip()
                            
                            # Update board data
                            self.boards_data[board_id]['last_message'] = message
                            self.boards_data[board_id]['last_seen'] = time.time()
                            
                            print(f"Received from Board {board_id}: {message}")
                        except Exception as e:
                            print(f"Error parsing advertisement: {e}")
            
            # Clean up
            proc.terminate()
                    
        except Exception as e:
            print(f"Error monitoring advertisements: {e}")
        finally:
            # Make sure to clean up the process
            if proc:
                try:
                    proc.terminate()
                except:
                    pass
    
    def stop_scanning(self):
        self.running = False
        if self.scan_process:
            self.scan_process.terminate()
            self.scan_process = None

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
            # Example of sending to all boards
            if message_counter % 5 == 0:
                message = f"ALL:{message_counter}"
                broadcast_message(advertiser, message, target_id=0)  # 0 = broadcast to all
            
            # Example of sending to individual boards
            else:
                target_board = (message_counter % 3) + 1  # Boards 1, 2, 3
                # message = f"MSG:{message_counter}"
                message = "IM THE PC "
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