import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib
import array
import struct
import time
import threading

# Constants
BLUEZ_SERVICE_NAME = 'org.bluez'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
LE_ADVERTISEMENT_IFACE = 'org.bluez.LEAdvertisement1'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHRC_IFACE = 'org.bluez.GattCharacteristic1'

# UUID for broadcast service
BROADCAST_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
# UUID for receiving data from boards
RECEIVE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

mainloop = None

class BLEAdvertisement(dbus.service.Object):
    def __init__(self, bus, index):
        self.path = f"/org/bluez/example/advertisement{index}"
        self.bus = bus
        self.ad_type = "peripheral"
        self.service_uuids = None
        self.manufacturer_data = None
        self.solicit_uuids = None
        self.service_data = None
        self.local_name = None
        dbus.service.Object.__init__(self, bus, self.path)

    def set_manufacturer_data(self, manuf_code, data):
        if self.manufacturer_data is None:
            self.manufacturer_data = dbus.Dictionary({}, signature='qv')
        self.manufacturer_data[manuf_code] = dbus.Array(data, signature='y')

    def set_local_name(self, name):
        self.local_name = name

    def add_service_uuid(self, uuid):
        if self.service_uuids is None:
            self.service_uuids = []
        self.service_uuids.append(uuid)

    def add_service_data(self, uuid, data):
        if self.service_data is None:
            self.service_data = dbus.Dictionary({}, signature='sv')
        self.service_data[uuid] = dbus.Array(data, signature='y')

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_IFACE:
            raise dbus.exceptions.InvalidArguments("Invalid interface")
        
        properties = dict()
        properties["Type"] = self.ad_type
        
        if self.service_uuids is not None:
            properties["ServiceUUIDs"] = dbus.Array(self.service_uuids, signature='s')
        if self.manufacturer_data is not None:
            properties["ManufacturerData"] = self.manufacturer_data
        if self.service_data is not None:
            properties["ServiceData"] = self.service_data
        if self.local_name is not None:
            properties["LocalName"] = self.local_name
            
        return properties

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature='', out_signature='')
    def Release(self):
        print("Advertisement released")

class Service(dbus.service.Object):
    def __init__(self, bus, index, uuid, primary):
        self.path = f"/org/bluez/example/service{index}"
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.InvalidArguments("Invalid interface")
        
        properties = dict()
        properties["UUID"] = self.uuid
        properties["Primary"] = self.primary
        properties["Characteristics"] = dbus.Array(
            [characteristic.path for characteristic in self.characteristics],
            signature='o')
            
        return properties

class Characteristic(dbus.service.Object):
    def __init__(self, bus, index, uuid, flags, service):
        self.path = f"{service.path}/char{index}"
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.value = [0]
        dbus.service.Object.__init__(self, bus, self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise dbus.exceptions.InvalidArguments("Invalid interface")
        
        properties = dict()
        properties["UUID"] = self.uuid
        properties["Service"] = self.service.path
        properties["Flags"] = self.flags
        properties["Value"] = self.value
            
        return properties

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options):
        print("Reading characteristic value")
        return self.value

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='aya{sv}', out_signature='')
    def WriteValue(self, value, options):
        print(f"Received data from a MicroPython board: {bytes(value)}")
        self.value = value

class ReceiverCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index,
            RECEIVE_CHAR_UUID,
            ["write", "write-without-response"],
            service)

    def WriteValue(self, value, options):
        board_data = bytes(value)
        print(f"Received from board: {board_data.decode('utf-8', errors='replace')}")
        self.value = value

class BroadcastService(Service):
    def __init__(self, bus, index):
        Service.__init__(self, bus, index, BROADCAST_SERVICE_UUID, True)
        self.add_characteristic(ReceiverCharacteristic(bus, 0, self))

def find_adapter(bus):
    remote_om = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, '/'),
        DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    
    for o, props in objects.items():
        if LE_ADVERTISING_MANAGER_IFACE in props and GATT_MANAGER_IFACE in props:
            return o
    
    raise Exception("No BLE adapter found with both advertising and GATT capabilities")

def register_advertisement(bus, advertisement, adapter_path):
    advertisement_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
        LE_ADVERTISING_MANAGER_IFACE)
    # Change this line from {} to {"":""} to avoid empty dict error
    advertisement_manager.RegisterAdvertisement(
        advertisement.path, {"":""})

def register_services(bus, services, adapter_path):
    gatt_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
        GATT_MANAGER_IFACE)
    for service in services:
        # Change this line from {} to {"":""} to avoid empty dict error
        gatt_manager.RegisterService(service.path, {"":""})

def update_advertisement_data(advertisement, message):
    # Clear previous data
    advertisement.manufacturer_data = None
    # Convert message to bytes and use manufacturer-specific data
    # Using manufacturer code 0xFFFF (for development/testing)
    data = [ord(c) for c in message]
    advertisement.set_manufacturer_data(0xFFFF, data)
    
    # Also update service data for devices that might be scanning for it
    advertisement.service_data = None
    advertisement.add_service_data(BROADCAST_SERVICE_UUID, data)

def broadcaster_thread(advertisement):
    message_counter = 0
    while True:
        # Create a message to broadcast
        message = f"Broadcast #{message_counter} @ {time.time():.2f}"
        print(f"Broadcasting: {message}")
        
        # Update the advertisement with the new message
        update_advertisement_data(advertisement, message)
        
        # Sleep for a while before sending the next message
        time.sleep(1)
        message_counter += 1

def main():
    global mainloop
    
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    
    # Get the adapter
    adapter_path = find_adapter(bus)
    print(f"BLE adapter found: {adapter_path}")
    
    # Create and register service
    service = BroadcastService(bus, 0)
    services = [service]
    register_services(bus, services, adapter_path)
    
    # Create an advertisement
    advertisement = BLEAdvertisement(bus, 0)
    advertisement.add_service_uuid(BROADCAST_SERVICE_UUID)
    advertisement.set_local_name("PC-Broadcaster")
    
    # Initial message
    initial_message = "PC BLE Broadcaster Ready"
    update_advertisement_data(advertisement, initial_message)
    
    # Register advertisement
    register_advertisement(bus, advertisement, adapter_path)
    print("Advertisement registered")
    
    # Start a thread to update the advertisement periodically
    broadcast_thread = threading.Thread(target=broadcaster_thread, args=(advertisement,))
    broadcast_thread.daemon = True
    broadcast_thread.start()
    
    # Run the main loop
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == '__main__':
    main()