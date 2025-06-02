import sys
import numpy as np

from Env import env_v1
from PSO import PSO_v1
from PSO import PSO_v2
from Env.agents_v1 import follow_path_wp 



# General Settings
not_env_flag = False


# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (50, 550), #(450, 550), #(50, 400), #(50, 550),
    'num_agents': 3,
    'formation_type': 2
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    # 'num_obs': 30,
    'num_obs': 10,
    # 'type_obs': 'random',                  # Simple Map Grid
    'type_obs': 'warehouse_0',                  # More elements Map Grid
    'max_rect_obs_size': 200,                   # maximun Obstacle size
    'seed_val_obs': 80, # 286                   # Test obstacles location
    'mouse_flag': True                          # Mouse pointer is turned in a sqaere obstacle
}

# 860

# PSO Settings
pso_params = {
    'iterations': 100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.7, #0.2,
    'Cg': 0.1,
    'num_particles': 100,
    'resolution': 10
}




# Initialize Environment
obst_list = []

if not not_env_flag :
    target_pos = [(1100, 100)]
    env = env_v1.Environment(map_settings, agents_settings)
    env.initialize()

    obst_list = env.env_map.random_rect_obs_list
else:
    # Manual Obstacles
    # x_botton, y_botton, width, height
    # obst_list.append((400, 200, 200, 200))
    # target_pos = [(900, 100)] 
    

    # Tested
    obst_list.append((400, 200, 400, 300))    
    target_pos = [(600, 550), (1100, 100)]  # resolution 5 (start point = (50, 550)

    # # Tested
    # obst_list.append((400, 200, 400, 300))    
    # target_pos = [(1100, 100)]              # resolution 2 (start point = (50, 550)

    # Tested
    # obst_list.append((400, 200, 400, 300))    # start point (50, 400)
    # target_pos = [(1100, 280)]  # resolution 2

    # Tested
    # obst_list.append((400, 200, 400, 300))    
    # target_pos = [(600, 150)]  # resolution 2 (450, 550)
    
    

# Compute the PSO Path 
pso_item = PSO_v1.PSO(map_settings['map_dimensions'], agents_settings['start_pos'], target_pos, pso_params, obst_list)
# pso_item = PSO_v2.PSO(map_settings['map_dimensions'], agents_settings['start_pos'], target_pos, pso_params, env.env_map.random_rect_obs_list)
pso_item.pso_compute()


pso_item.visualization()

# pso_item.collision_rect_lastCorrection(pso_item.G[0, :])
pso_item.collision_rect_lastCorrection_v2(pso_item.G[0, :])
pso_item.visualization_lastAdjustment()
# pso_item.visualization_all()


pso_item.visualization_sequence()




sys.exit()