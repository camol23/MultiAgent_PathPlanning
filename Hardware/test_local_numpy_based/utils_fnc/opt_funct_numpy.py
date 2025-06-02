#!/usr/bin/env python3

import numpy as np

def data_state_distance(states):
    '''
        states : (iteretaions, agent, (x,y))
    '''

    x_diff = states[:-1, :, 0] - states[1:, :, 0]
    y_diff = states[:-1, :, 1] - states[1:, :, 1]

    x_power = x_diff**2
    y_power = y_diff**2
    xy_sum = x_power + y_power

    x_sqrt = np.sqrt(xy_sum)
    dist_total = np.sum(x_sqrt)

    return dist_total


def density_map(map_size, obst_known, obst_unknown):
    '''
        density =  obst_area / map_area
    '''

    map_area = map_size[0]*map_size[1]

    obst_known_area = 0
    for obst in obst_known:
        obst_known_area = obst_known_area + obst[2]*obst[3]

    obst_unknown_area = 0
    for obst in obst_unknown:
        obst_unknown_area = obst_unknown_area + obst[2]*obst[3]

    density = (obst_known_area + obst_unknown_area)/map_area

    return density
    