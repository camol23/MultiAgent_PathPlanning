#!/usr/bin/env python3



def state_interpreter(data, vis=False):
    data = data.split(",")

    vel_right = int(data[0])
    vel_left = int(data[1])

    # Vis.
    if vis :
        print("Vel. from robot [deg/s] = ", vel_right, vel_left)
        

    return vel_right, vel_left


def state_interpreter2(data, vis=False):
    data = data.split(",")

    vel_right = int(data[0])
    vel_left = int(data[1])
    sensor_dist = float(data[2])


    # Vis.
    if vis :
        print("Vel. from robot [deg/s] = ", vel_right, vel_left)
        print("Sensor  Distance = ", sensor_dist)

    return vel_right, vel_left, sensor_dist