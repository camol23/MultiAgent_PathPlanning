#!/usr/bin/env python3

import rpyc
import time


# Ref:
# (1) https://ev3dev-lang.readthedocs.io/projects/python-ev3dev/en/stable/rpyc.html
# (2) https://ev3dev-lang.readthedocs.io/projects/python-ev3dev/en/stable/sensors.html#ultrasonic-sensor

# Create a RPyC connection to the remote ev3dev device.
# Use the hostname or IP address of the ev3dev device.
# If this fails, verify your IP connectivty via ``ping X.X.X.X``
conn_1 = rpyc.classic.connect('10.16.12.26')
conn_2 = rpyc.classic.connect('10.16.12.25')

# import ev3dev2 on the remote ev3dev device
ev3dev2_1_motor = conn_1.modules['ev3dev2.motor']
ev3dev2_2_motor = conn_2.modules['ev3dev2.motor']

# ev3dev2_sensor = conn.modules['ev3dev2.sensor']
# ev3dev2_sensor_lego = conn.modules['ev3dev2.sensor.lego']

ev3dev2_1_sensor = conn_1.modules['ev3dev2.sensor']
ev3dev2_1_sensor_lego = conn_1.modules['ev3dev2.sensor.lego']

ev3dev2_2_sensor = conn_2.modules['ev3dev2.sensor']
ev3dev2_2_sensor_lego = conn_2.modules['ev3dev2.sensor.lego']

# Use the LargeMotor and TouchSensor on the remote ev3dev device
motor_1 = ev3dev2_1_motor.LargeMotor(ev3dev2_1_motor.OUTPUT_C)
motor_2 = ev3dev2_2_motor.LargeMotor(ev3dev2_2_motor.OUTPUT_C)

# ts = ev3dev2_sensor_lego.TouchSensor(ev3dev2_sensor.INPUT_1)
dist_sensor1_1 = ev3dev2_1_sensor_lego.UltrasonicSensor(ev3dev2_1_sensor.INPUT_2)
dist_sensor2_1 = ev3dev2_2_sensor_lego.UltrasonicSensor(ev3dev2_2_sensor.INPUT_2)

# If the TouchSensor is pressed, run the motor

time_run_1 = time.time()
motor_1.run_forever(speed_sp=200)
time_run_1 = time.time() - time_run_1
time_run_2 = time.time()
motor_2.run_forever(speed_sp=200)
time_run_2 = time.time() - time_run_2
# motor_2.stop()
# speed = self._current_movement['speed']
# while True:
read_speed = []
read_sensor = []
iter = 20
for i in range(0, iter):
    # motor.run_forever(speed_sp=200)
    time_speed = time.time()
    print(motor_1.speed, motor_2.speed)
    time_speed = time.time() - time_speed

    time_dist = time.time()
    distance_1 = dist_sensor1_1.distance_centimeters
    time_dist = time.time() - time_dist

    read_speed.append( time_speed )
    read_sensor.append(time_dist)    

    distance_2 = dist_sensor2_1.distance_centimeters
    print("Distance = ", distance_1, distance_2)
    time.sleep(0.5)



motor_1.stop()
motor_2.stop()

sum_speed = 0
for i, t in enumerate(read_speed):
    sum_speed = t + sum_speed

speed_mean = sum_speed/(i+1)

sum_dist = 0
for i, t in enumerate(read_sensor):
    sum_dist = t + sum_dist

dist_mean = sum_dist/(i+1)


print()
print("Time Data")
print("Ave. T speed ", speed_mean, time_speed)
print("Ave. T dist ", dist_mean, time_dist)
print("T run ev1", time_run_1)
print("T run ev2", time_run_2)