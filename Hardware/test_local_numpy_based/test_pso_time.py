#!/usr/bin/env python3

import time
import numpy as np

from PSO import PSO_v1
from Env import env_small
from logging_data import store_data


num_iter = 20 #100

# Map Settings
map_size = (200, 200)

start_pos = (5, int(map_size[1]/2))
target_pos = [(int(map_size[0]-5), int(map_size[1]/2))]  

# Settings
num_samples = 3 # 5
num_scenarios = 10

# Variable to Evaluate
resolution_list = [5, 10]
number_obst = [((2*i)+1) for i in range(0, num_scenarios)]
# number_obst = [16]
max_rect_obs_size = 30


# Storage 
pso_time_mean = np.zeros((num_scenarios, len(resolution_list)) )
pso_time_std = np.zeros((num_scenarios, len(resolution_list)) )


# PSO Settings
pso_params = {
    'iterations': num_iter,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.7, #0.2,
    'Cg': 0.1,
    'num_particles': 100, #30, #50, #100
    'resolution': 5
}

for scenarios_idx in range(0, num_scenarios):
    # Env.
    obst_list = env_small.random_obstacles(number_obst[scenarios_idx], map_size[0], map_size[1], max_rect_obs_size)

    # Data
    round_time_list  = []

    for r_idx in range(0, len(resolution_list)):
        pso_params['resolution'] = resolution_list[r_idx]

        for sample_idx in range(0, num_samples):    
            # Compute the PSO Path 
            pso_item = PSO_v1.PSO(map_size, start_pos, target_pos, pso_params, obst_list)
            start_time = time.time()
            pso_item.pso_compute()
            end_time = time.time() - start_time

            # pso_item.visualization()
            # pso_item.collision_rect_lastCorrection_v2(pso_item.G[0, :])
            # pso_item.visualization_lastAdjustment()
            # pso_item.visualization_sequence()

            round_time_list.append( end_time )


        # Statistics
        round_time = np.array(round_time_list)*100
        round_time_mean = np.mean(round_time)
        round_time_std = np.std(round_time)

        # Storage
        pso_time_mean[scenarios_idx, r_idx] = round_time_mean
        pso_time_std[scenarios_idx, r_idx] = round_time_std


# print(pso_time_mean)


# Prepaer Data to be saved
dict_data = {
    'pso_time_mean' : pso_time_mean,
    'pso_time_std' : pso_time_std,
    'number_obst' : number_obst,
    'resolution_list' : resolution_list
}

# path = "./logging_data/time_data_exp/"
path = '/home/robot/test_local_numpy_based/logging_data/time_data_exp/'
name = "pso_routes_ev3_time_test_iter_" + str(num_iter)
store_data.save_pickle(name, dict_data, path)

