from ev3dev2.motor import OUTPUT_A, OUTPUT_B, MoveTank, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor

# Instantiate the MoveTank object
tank = MoveTank(OUTPUT_A, OUTPUT_B)

# Initialize the tank's gyro sensor
tank.gyro = GyroSensor()

# Calibrate the gyro to eliminate drift, and to initialize the current angle as 0
tank.gyro.calibrate()

# Pivot 30 degrees
tank.turn_degrees(
    speed=SpeedPercent(5),
    target_angle=30
)