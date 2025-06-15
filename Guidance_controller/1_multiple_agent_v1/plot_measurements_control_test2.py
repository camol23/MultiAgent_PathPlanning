from logging_data import store_data

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils_fnc import op_funct


file_path = "./logging_data/measurements/"
file_name = file_path + "control_ekf_test"
test_dict = store_data.load_time_data_pickle(file_name)


state_list = test_dict['state_list']
state_ekf_list = test_dict['state_ekf_list']
policy_Tr = test_dict['policy_Tr']


# Results 
fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_list]
state_y = [x[1] for x in state_list]

state_ekf_x = [x[0] for x in state_ekf_list]
state_ekf_y = [x[1] for x in state_ekf_list]

trajectory_x = [val[0] for val in policy_Tr]
trajectory_y = [val[1] for val in policy_Tr]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 
ax.plot(state_ekf_x, state_ekf_y, color ='green') 

ax.grid(True)
ax.axis('equal')
plt.show() 


fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [i for i in range(0, len(state_list))]
# state_y = [op_funct.radToDeg(x[2]) for x in state_list]
state_y = [x[2] for x in state_list]

state_ekf_x = [i for i in range(0, len(state_ekf_list))]
state_ekf_y = [op_funct.radToDeg(x[2]) for x in state_ekf_list]

# print("")

ax.plot(state_x, state_y, color ='blue') 
ax.plot(state_ekf_x, state_ekf_y, color ='green') 

ax.grid(True)
ax.axis('equal')
plt.show() 