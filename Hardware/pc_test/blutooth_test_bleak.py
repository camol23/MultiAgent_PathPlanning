import asyncio
from bleak import BleakScanner, BleakClient

async def bleak_receive():
    """
    Receive data using Bleak (modern, cross-platform)
    Better for newer Bluetooth implementations
    """
    def on_data_received(sender, data):
        """Callback for received data"""
        print(f"Received data: {data}")
    
    # Scan for devices
    devices = await BleakScanner.discover()
    
    # Print discovered devices (helpful for finding the right address)
    for device in devices:
        print(f"Device: {device.address} - {device.name}")
    
    # Replace with the actual EV3 device address you want to connect to
    ev3_device_address = "F0:45:DA:11:92:74"  # Your EV3's Bluetooth address
    
    # Characteristic UUID (this will depend on your specific Bluetooth service)
    characteristic_uuid = "00001101-0000-1000-8000-00805f9b34fb"
    
    async with BleakClient(ev3_device_address) as client:
        # Start notifications on a specific characteristic
        await client.start_notify(characteristic_uuid, on_data_received)
        
        # Keep connection open
        await asyncio.sleep(60)  # Keep running for 60 seconds

# Main execution method
def main():
    # Run the async function
    asyncio.run(bleak_receive())

# Proper way to call the function
if __name__ == "__main__":
    main()