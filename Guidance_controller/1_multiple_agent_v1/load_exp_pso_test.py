from logging_data import store_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors




units_factor = 1000_000    
file_path = "./logging_data/time_test/"
# name = 'pso_routes_ev3_time_test_iter_100'
# name = 'pso_routes_ev3_time_test_iter_40_p50'
name = 'pso_routes_ev3_time_test_iter_80_p30'
name = 'pso_routes_ev3_time_test_iter_20_p100'
file_name = file_path + name

time_dict = store_data.load_time_data_pickle(file_name)


resolution_list = time_dict['resolution_list']
number_obst = time_dict['number_obst']
pso_time_mean = time_dict['pso_time_mean']
pso_time_std = time_dict['pso_time_std']


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

