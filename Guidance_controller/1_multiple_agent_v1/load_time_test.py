from logging_data import store_data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Define x-axis
order_type = 0      # 0: Density
                    # 1: Num. Obstacles

units_factor = 1000_000    

# load_scene_params['scene_name'] +"_method_"+ remap_persistance_flag
file_name1_list = ['scene_obs_0_a5_method_2', 'scene_obs_2_a5_method_2', 'scene_obs_3_a5_method_2', 'scene_2_obs_5_a5_method_2', 'scene_2_obs_7_a5_method_2']      # remap measures
file_name2_list = ['scene_obs_0_a5_method_4', 'scene_obs_2_a5_method_4', 'scene_obs_3_a5_method_4', 'scene_2_obs_5_a5_method_4', 'scene_2_obs_7_a5_method_4']      # Obst. avoidance
file_path = "./logging_data/time_test/"

time_data_list = []
execution_time_list = []
execution_time_std_list = []
density_list = [] 
num_total_obst = []
num_obst_unknowns = []

experinents_list = [file_name1_list, file_name2_list] # Lines plots
execution_time_sorted_list = []
execution_time_std_sorted_list = []
density_list_sorted_list = []
num_total_obst_sorted_list = []
num_total_obst_unknown_sorted_list = []

max_time = 0
for file_name_list in experinents_list :

    # Re-start points arrays
    execution_time_list = []
    execution_time_std_list = []
    density_list = [] 
    num_total_obst = []
    num_obst_unknowns = []
    # Gather points for each Experiment for Type *
    for name in file_name_list:
        file_name = file_path + name
        time_dict = store_data.load_time_data_pickle(file_name)

        # time_data_list.append( time_dict )
        exe_time_array = time_dict["execution_time"]*units_factor       
        execution_time_list.append( np.mean(exe_time_array) )
        execution_time_std_list.append( np.std(exe_time_array) )
        density_list.append( time_dict["map_density"] )
        num_total_obst.append( time_dict["num_obst"] )
        num_obst_unknowns.append(time_dict["num_obst_unknowns"] )

        max_temp = np.max(exe_time_array)
        if max_temp > max_time :
            max_time = max_temp
        # print(np.mean(exe_time_array))
        # print(np.max(exe_time_array, axis=0))
        # print(np.std(exe_time_array, axis=0))
        # print()


    # Sort by Num_obst or Density (x-axis)
    index_sorted = []
    if order_type :
        num_total_obst = np.array(num_total_obst)
        index_sorted = np.argsort(num_total_obst) 
    else:
        density_list = np.array(density_list)
        index_sorted = np.argsort(density_list) 

    
    # (y-axis)
    execution_time_sorted = [execution_time_list[i] for i in index_sorted]
    execution_time_std_sorted = [execution_time_std_list[i] for i in index_sorted]
    # (x-axis)
    density_list_sorted = [density_list[i] for i in index_sorted]
    num_total_obst_sorted = [num_total_obst[i] for i in index_sorted]
    num_obst_unknowns_sorted = [num_obst_unknowns[i] for i in index_sorted]

    execution_time_sorted_list.append( execution_time_sorted )
    execution_time_std_sorted_list.append( execution_time_std_sorted )
    density_list_sorted_list.append( density_list_sorted )
    num_total_obst_sorted_list.append( num_total_obst_sorted )
    num_total_obst_unknown_sorted_list.append( num_obst_unknowns_sorted )


print(num_total_obst_sorted_list)
print(density_list_sorted_list)
print("STD [us] = ", execution_time_std_sorted_list[0])
print("Max STD [us] = ", np.max(execution_time_std_sorted_list[0]))
print("Max Ave [us] = ", np.max(execution_time_sorted_list[0]))
print("Max Time [us] = ", max_time)
# Ploting
colors = [
    'rosybrown',
    'steelblue',
    'mediumpurple'
]


fig, axes = plt.subplots(1, 1)

titles = ['Remapping', 'Obst. Avoidance']
x_axis_name = ['map density [%]', 'number of Obstacles']
if order_type :
    delta_name = [0, 10] # num Obst
else:
    delta_name =  [0.5, 1] # Density
    delta_name2 = [0.5, 20] # Density
    # name_add = ['O  ', 'UO ']
for i in range(0, 2):

    # color_i = mcolors.CSS4_COLORS[ploting.colors_agent[i]]
    color_i = mcolors.CSS4_COLORS[colors[i]]
    if order_type: 
        x_axis_data = num_total_obst_sorted_list[i]        
        text_name = np.round(np.array(density_list_sorted_list[i])*100) 
    else:
        # x_axis_data = density_list_sorted_list[i]
        x_axis_data = np.array(density_list_sorted_list[i])*100
        text_name = num_total_obst_unknown_sorted_list[i]

    axes.plot(x_axis_data, execution_time_sorted_list[i], color =color_i, alpha=0.7, label=titles[i]) 
    axes.scatter(x_axis_data, execution_time_sorted_list[i], color=color_i, alpha=0.5, linewidths=0.2)
    
    # axes.errorbar(x_axis_data, execution_time_sorted_list[i], yerr=execution_time_std_sorted_list[i], fmt='o', color=color_i,
    #                 ecolor=color_i, elinewidth=2, capsize=0, alpha=0.4)

    for j, x_text in enumerate(x_axis_data): 
        axes.text(x_text+delta_name[0], execution_time_sorted_list[i][j]+delta_name[1], str(text_name[j]), fontsize=8, horizontalalignment='left',
                        verticalalignment='center', color=mcolors.CSS4_COLORS['grey'])
        # axes.text(x_text+delta_name2[0], execution_time_sorted_list[i][j]+delta_name2[1], name_add[1] + str(text_name[j]), fontsize=8, horizontalalignment='left',
        #                 verticalalignment='center', color=mcolors.CSS4_COLORS['grey'])


axes.grid(which="major", color="0.9")
axes.legend() # loc=2
axes.set_xlabel(x_axis_name[order_type])
axes.set_ylabel('Time [us]')
# axes.grid(True)
# ax.axis('equal')
plt.show() 