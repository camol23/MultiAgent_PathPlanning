import asyncio
from bleak import BleakScanner

async def discover_devices():
    """
    Scan for and print all nearby Bluetooth devices
    """
    print("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()
    
    print("\nDiscovered Devices:")
    for device in devices:
        print(f"Address: {device.address}")
        print(f"Name: {device.name}")
        print("---")

# Run the discovery
asyncio.run(discover_devices())


