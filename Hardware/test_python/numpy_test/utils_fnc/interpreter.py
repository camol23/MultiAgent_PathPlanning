#!/usr/bin/env python3

import copy 


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


def change_units(init_agent, target_routes, obst_list, obst_list_unkowns, map_size, data_lists, divider=10):

    init_agent = [[val[0]/divider, val[1]/divider] for val in init_agent]
    obst_list = [[val[0]/divider, val[1]/divider, val[2]/divider, val[3]/divider] for val in obst_list]
    obst_list_unkowns = [[val[0]/divider, val[1]/divider, val[2]/divider, val[3]/divider] for val in obst_list_unkowns]
    map_size = [map_size[0]/divider, map_size[1]/divider]

    new_target = []
    for target in target_routes:
        new_target.append( [(val[0]/divider, val[1]/divider) for val in target] )

    print()
    print("Target Routes")
    print(target_routes)
    target_routes = copy.deepcopy(new_target)
    print(target_routes)
    print()

    new_datalist = []
    print("data update")

        
    for route in data_lists :
        new_datalist.append( route/divider )

    print(data_lists)
    print()
    data_lists = copy.deepcopy(new_datalist)
    print(data_lists)


    return init_agent, target_routes, obst_list, obst_list_unkowns, map_size, data_lists