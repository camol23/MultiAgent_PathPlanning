from ev3dev2.motor import OUTPUT_A, OUTPUT_B, MoveTank, SpeedPercent, follow_for_ms
from ev3dev2.sensor.lego import GyroSensor

# Ref:https://github.com/ev3dev/ev3dev-lang-python/blob/ev3dev-stretch/ev3dev2/motor.py

# Instantiate the MoveTank object
tank = MoveTank(OUTPUT_A, OUTPUT_B)

# Initialize the tank's gyro sensor
tank.gyro = GyroSensor()

# Calibrate the gyro to eliminate drift, and to initialize the current angle as 0
tank.gyro.calibrate()

try:

    # Follow the target_angle for 4500ms
    tank.follow_gyro_angle(
        kp=11.3, ki=0.05, kd=3.2,
        speed=SpeedPercent(30),
        target_angle=0,
        follow_for=follow_for_ms,
        ms=4500
    )
except FollowGyroAngleErrorTooFast:
    tank.stop()
    raise