
import numpy as np
import math

from Env import env_small
from logging_data import store_data
from Aux_libs import ploting
from utils_fnc import interpreter


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.animation as animation


# Succesfull exp. test:
    #   (1) run_data_6_8_17_16_agent_0
    #   (2) run_data_6_8_17_36_agent_1

PATH = '/home/camilo/Documents/experimental_test/'
file_name = PATH + 'run_data_6_8_17_36_agent_0' # 'run_data_5_15_8_45_agent_0'
file_name_a2 = PATH + 'run_data_6_8_17_36_agent_1' # 'run_data_5_15_8_45_agent_1'



divider_units = 100
Ts = 0.4

# Small Environment
load_scene_params = {
    # 'scene_name' : 'scene_0_a5'
    # 'scene_name' : 'scene_0_a2'

    'scene_name' : 'scene_experiment_test0_a2'
    # 'scene_name' : 'scene_0_a2_exp_line'

    # 'scene_name' : 'scene_obs_0_a5'
    # 'scene_name' : 'scene_force_collision_obs'
    # 'scene_name' : 'scene_force_collision_obs_a1'
}

# Agents
init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)

# Take Obstacle list
obst_original = ploting.convert_obs_coor(obst_list)
obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)


# Units converted (REQUIRE)
init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, _ = interpreter.change_units(init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, [], divider=divider_units)
_, _, obst_list, obst_list_unkowns, _, _ = interpreter.change_units(init_agent, target_routes, obst_list, obst_list_unkowns, map_size, [], divider=divider_units)

# -------------------------------------- Load Data -------------------------------------------------

data_a1 = store_data.load_runData_pickle(file_name)
data_a2 = store_data.load_runData_pickle(file_name_a2)

agents_data_list = [data_a1, data_a2]
data_jointed = store_data.joint_swarm_data(agents_data_list)


ploting.plot_jointed_data(data_jointed, target_routes, obst_original, obst_list_unkowns_converted, cm_flag=False, smaller=True)

scene_axes, state_lines, figure, axes = ploting.animate_scene_jointed_data(data_jointed, target_routes, obst_original, obst_list_unkowns_converted, cm_flag=False, smaller=True)

jump = 50

state_list = []
for data_dict in agents_data_list:
    
    state_storage_i = data_dict['state']
    state_i_x = [x[0] for x in state_storage_i]
    state_i_y = [x[1] for x in state_storage_i]

    state_list.append([state_i_x, state_i_y])

total_iter = len(state_storage_i)

def animate_update(frame):
    
    state_lines_2 = []
    for i, data_dict in enumerate(agents_data_list):

        brake_actived = data_dict['brake_actived']
        obs_detected = data_dict['obs_detected']        

        if brake_actived[frame*jump] :
            color_agent = 'black'

        elif obs_detected[frame*jump] :
            color_agent = 'tomato'

        else:
            color_agent = 'blue'
    
        state_lines_2.append( axes.plot(state_list[i][0][:frame*jump], state_list[i][1][:frame*jump], color= mcolors.CSS4_COLORS[color_agent]) )


            

    return scene_axes + state_lines_2


ani = animation.FuncAnimation(fig=figure, func=animate_update, frames=int(total_iter/jump), interval=Ts*1000)
# plt.tight_layout()
plt.show()