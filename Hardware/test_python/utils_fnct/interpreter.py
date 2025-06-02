#!/usr/bin/env pybricks-micropython



def policy_interpreter(data):
    data = data.split(",")

    # Vel.
    # data_right = min(200, float(data[0]) )
    # data_left = min(200, float(data[1]) )

    # No negative
    data_right = max( min(200, float(data[0]) ), 0)
    data_left = max( min(200, float(data[1]) ), 0)

    # Stop
    stop = int(data[2])

    return data_right, data_left, stop