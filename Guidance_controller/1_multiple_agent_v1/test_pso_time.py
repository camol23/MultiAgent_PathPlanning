
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PSO import PSO_v1
from Env import env_small
from Aux_libs import ploting





# Map Settings
map_size = (200, 200)

start_pos = (5, int(map_size[1]/2))
target_pos = [(int(map_size[0]-5), int(map_size[1]/2))]  

# Settings
num_samples = 5
num_scenarios = 1

# Variable to Evaluate
resolution_list = [12] #[5, 10, 20]
# number_obst = [2*i for i in range(0, num_scenarios)]
number_obst = [16]
max_rect_obs_size = 30


# Storage 
pso_time_mean = np.zeros((num_scenarios, len(resolution_list)) )
pso_time_std = np.zeros((num_scenarios, len(resolution_list)) )


# PSO Settings
pso_params = {
    'iterations': 300,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.7, #0.2,
    'Cg': 0.1,
    'num_particles': 100,
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

            pso_item.visualization()
            pso_item.collision_rect_lastCorrection_v2(pso_item.G[0, :])
            pso_item.visualization_lastAdjustment()
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



# Ploting

colors = [
    'rosybrown',
    'steelblue',
    'mediumpurple'
]

# fig = plt.figure() 
# ax = fig.add_subplot(1, 1, 1) 
fig, axes = plt.subplots(1, 1)

for i, resolution in enumerate(resolution_list):

    # color_i = mcolors.CSS4_COLORS[ploting.colors_agent[i]]
    color_i = mcolors.CSS4_COLORS[colors[i]]
    axes.plot(number_obst, pso_time_mean[:, i], color =color_i, alpha=0.7, label="resolution " +str(resolution)) 
    # axes.scatter(number_obst, pso_time_mean[:, i], color=color_i, alpha=0.5, linewidths=0.5)
    
    axes.errorbar(number_obst, pso_time_mean[:, i], yerr=pso_time_std[:, i], fmt='o', color=color_i,
                    ecolor='lightgray', elinewidth=3, capsize=0)

axes.grid(which="major", color="0.9")
axes.legend(loc=2)
axes.set_xlabel('Number of Obstacles')
axes.set_ylabel('Time [ms]')
# axes.grid(True)
# ax.axis('equal')
plt.show() 