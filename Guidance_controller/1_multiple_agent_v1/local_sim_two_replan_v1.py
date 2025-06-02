'''
    Just testing porpuses
'''


import numpy as np
import math

from PSO import pso_iterator
from PSO import PSO_decision
from PSO import PSO_auxiliar
from Aux_libs import ploting

from utils_fnc import op_funct, interpreter
from Env import env_small
from control import model, controllers, obs_avoidance, proximity_action
from logging_data import store_data

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.animation as animation


# ---------------------------------------------------------------------------------------------------
#                                           SETTINGS
# ---------------------------------------------------------------------------------------------------


agent_id = 0

remap_flag = False                           # True : The route is re-computed
remap_type = "Obstacle_detection" 
# remap_type = "new_data"                      # Update base on new Obstacles discovered
time_map_update = 1000                       # Iterationsto apply the remapping

total_iter = 15000                           # For loop Iteretions
Ts = 0.4                                     # Sample time

max_speed_sim = op_funct.deg2m(200, 0.056/2)
print("max_speed_sim ", max_speed_sim)
max_vel = max_speed_sim #0.001

divider_units = 100
tol_goal = 10/divider_units                                 # Stop signal when reach the goal [cm]


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

load_complementary_scene_params = {
    'scene_name' : 'scene_obs_0_a5_complement_1'
}

#
# PSO
#
Plot_from_PSO = False                       # Plot the individual results for each Path from PSO Algo.

# PSO Settings
safe_margin_obs_pso = 10 # [m]

pso_params = {
    'iterations': 100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': 5
}
random_seed = 10
# PSO Decision Making
pso_params_routes = {
    'iterations': 20, 
    'w': 0.04, # 0.04
    'Cp': 0.1, #0.2, # 0.7
    'Cg': 0.6, # 0.1 # 0.1
    'num_particles': 30
}

pso_params_map_update = {
    'iterations': 100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': 5
}

robot_prams = {
    'x_init' : 0,
    'y_init' : 0,
    'theta_init' : 0,

    'l_width' : 0.105,                      # meters
    'r_width' : 0.056/2,                    # meters
    'Ts' : Ts
}

pf_control_params = {
    'kv' : 0.1,               # Vel. Gain         (sim = 0.8) works = 0.1
    'kw' : 5,                 # Angular Vel. Gain (sim = 5) works = 5
    'k_rot' : 3.5,              # Heading Sensitive (sim = 5) works = 5

    # trajectory
    'Tr' : [],
    
    # Aux. 
    'l_width' : 0.105,        # robot width (0.105)
    'Ts' : Ts
}

circle_avoidance_params = {
    'R' : 15,
    'd_center' : 0,
    'mid_matgin' : 25 + 7        # 7: is the 4th part of the circle diameter       
}


obs_algorithm_params = {
    'obs_method' : None,
    'margin' : 25,               # Sensor distannce to activate the Algorithm    
}

detect_obs_params = {
    'angle_range' : math.radians(30), #45
    'ray_length' : 25/divider_units
}

# Linit Speed (swarm)
brake_limit_margin = 20/divider_units

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

# Agents
init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)
_, _, obst_list_added, obst_list_unkowns_added, _ = env_small.load_scene(load_complementary_scene_params)

# Take Obstacle list
obst_original = ploting.convert_obs_coor(obst_list)
obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)

num_agents = len(init_agent)

# -------------------------------------- Trajectory Comp. ------------------------------------------------------

# -------------------------------------- Agent 1 
#
# PSO Computation
#
agent_id = 0
init_agentn = init_agent[agent_id]

# Agent n
pso_iter_an = pso_iterator.PSO_wrapper()
pso_iter_an.initialization(map_size, init_agentn, target_routes, pso_params, obst_list)
pso_iter_an.safe_margin_obs = safe_margin_obs_pso
pso_iter_an.itereration(plot_flag=Plot_from_PSO)

print("agent 1 Dist. List ", pso_iter_an.dist_list)
print()
# Store possible Paths
# an_possible_paths.append( pso_iter_an.paths )


# -------------------------------------- Agent 2 

agent_id = 1
init_agentn = init_agent[agent_id]

# Agent n
pso_iter_a2 = pso_iterator.PSO_wrapper()
pso_iter_a2.initialization(map_size, init_agentn, target_routes, pso_params, obst_list)
pso_iter_a2.safe_margin_obs = safe_margin_obs_pso
pso_iter_a2.itereration(plot_flag=Plot_from_PSO)

print("agent 2 Dist. List ", pso_iter_a2.dist_list)
print()


# -------------------------------------- SHARE DATA 
# Each agent received a dist_list  

# Interpreter function for received data
# Convert to numpy 
data_list_a2 = pso_iter_a2.dist_list
print("Data list from PSO shape ", data_list_a2)


# dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function 
dist_matrix = np.array(pso_iter_an.dist_list).reshape((len(pso_iter_an.dist_list), 1))     
dist_an = np.array(data_list_a2).reshape((len(data_list_a2), 1))    
dist_matrix = np.hstack((dist_matrix, dist_an))


# --------------------------------------
#
# Decision Making
#
# Test PSO for Assigning routes task
pso_routes = PSO_decision.PSO()
pso_routes.initialization(pso_params_routes, dist_matrix)
pso_routes.pso_compute()

print("PSO Route")
print(pso_routes.output_list)
print(pso_routes.output_routes_ids)
print("Dist. Cost= ", pso_routes.total_dist([pso_routes.output_list]))
print()


# Checking ...
print("Reference sol. ")
print(pso_routes.ref_solution)
print("Dist. Cost= ", pso_routes.total_dist([pso_routes.ref_solution]))
print()

# -------------------------------------- Local Method
print("Local Method")
pso_routes = PSO_decision.PSO()
traditional_path = pso_routes.assign_paths(dist_matrix)
print("Output ", traditional_path)

# Route ID 
pso_routes.output_list = traditional_path
pso_routes.routes2Ids()
traditional_path_route_id = pso_routes.output_routes_ids
print("Converted ", traditional_path_route_id)
# pso_routes.dist_matrix = dist_matrix
# print("Dist. Cost Local= ", pso_routes.total_dist([traditional_path]))
print()


# -------------------------------------- Agent 1 
agent_id = 0
target_route_id = traditional_path_route_id[agent_id]
selected_path = pso_iter_an.paths[target_route_id]


# -------------------------------------- Agent 2
agent_id = 1
target_route_id_a2 = traditional_path_route_id[agent_id]
selected_path_a2 = pso_iter_a2.paths[target_route_id_a2]

data_lists = [selected_path, selected_path_a2]

# Units converted (REQUIRE)
init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, data_lists = interpreter.change_units(init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, data_lists, divider=divider_units)
selected_path = data_lists[0] 
selected_path_a2 = data_lists[1] 
_, _, obst_list, obst_list_unkowns, _, _ = interpreter.change_units(init_agent, target_routes, obst_list, obst_list_unkowns, map_size, [], divider=divider_units)

ploting.plot_scene([i for i in range(0, 2)], data_lists, obst_original, target_routes, cm_flag=True, obs_unknowns=obst_list_unkowns_converted, smaller=True)



# -------------------------------------- Controller -------------------------------------------------

# Model
# -------------------------------------- Agent 1 
agent_id = 0

robot_prams['x_init'] = init_agent[agent_id][0]
robot_prams['y_init'] = init_agent[agent_id][1]
robot = model.model_diff_v1()
robot.initialize(robot_prams)
state = [robot.x, robot.y, robot.theta]


# Obstacle avoidance Tr.
obs_avoid = obs_avoidance.circle_avoidance()
obs_avoid.initialize(circle_avoidance_params)
# obs_avoid.angles = [180-a for a in range(15, 180, 15)]

# Obs. ALgorithm
obs_algorithm = obs_avoidance.obs_algorithm()
obs_algorithm_params['obs_method'] = obs_avoid
obs_algorithm.initialize(obs_algorithm_params, agent_id)

# Obstacle Detected
obs_detected = []

# Controller
# Keep in Mind: 
#   (*) dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]

# Prepare Trajectory for Controller
track = selected_path
print(" POLICY")
print(track)
trajectory = [[track[0][i].item(), track[1][i].item()] for i in range(0, len(track[0]))]
print(trajectory)
print()
pf_control_params['Tr'] = trajectory
policy = controllers.path_follower(pf_control_params)


# -------------------------------------- Map Update -------------------------------------------------
route_update = PSO_auxiliar.pso_map_update()
route_update.safe_margin_obs = safe_margin_obs_pso/divider_units
re_map_activation = 0


# -------------------------------------- Communication -------------------------------------------------
# Should Received Current Position Others
states_group = np.zeros((num_agents, 3))
states_group[1] = [init_agent[1][0], init_agent[1][1], 0]

# -------------------------------------- Simulaton -------------------------------------------------

# Store Vals.
# Dim : (sample_num, agent_idx, state[x,y,t])
state_storage = []
state_storage.append( state )

# Vr and Vl
vel_storage = [[0, 0]]

# Brake Plot Aux.
brake_actived = []


# Stop Signal
stop_agent_flag = 0

last_iteration_i = total_iter 
for i in range(0, total_iter):
    
    # print("Policy ", policy.Tr)
    vel_right = vel_storage[i][0]
    vel_left = vel_storage[i][1]
    # print(vel_right, vel_left)

    # Applying control signal & Read Response
    robot.step(vel_right, vel_left)                                                     # model step
    state = [robot.x, robot.y, robot.theta]    

    # Read Swarm Position
    states_group[0] = state
    states_group[1] = [init_agent[1][0], init_agent[1][1], 0]    

    # Brake Gradually
    brake_factor, active_brake = proximity_action.limit_speed(state, states_group, agent_id, margin=brake_limit_margin)
    brake_actived.append( active_brake )

    # Obstacle detection
    obs_detection_flag = obs_avoidance.detect_obs_sim(state, obst_list_unkowns, detect_obs_params)
    obs_detected.append( obs_detection_flag )

    # Obstacle Avoidance
    if obs_detection_flag :
        # Emulate sensor
        sensor_dist = 25
    else:
        sensor_dist = 50

    # Collision Avoidance ALgorithm
    # obs_algorithm.check_sensor(sensor_dist, policy.idx, state, policy.Tr)
    # if obs_algorithm.controller_update :
    #     policy.Tr = obs_algorithm.Tr_obs            
    #     policy.idx = obs_algorithm.idx_output


    # Re-Map
    if remap_type == "Obstacle_detection" :
        remap_flag = obs_detection_flag
    elif remap_type == "new_data" :
        if time_map_update == i :
            remap_flag = True
        else:
            remap_flag = False

    if remap_flag and (re_map_activation==0):
        if remap_type == "Obstacle_detection" :
            new_obst_list = obst_list + obst_list_unkowns 
        elif remap_type == "new_data" :
            new_obst_list = obst_list + obst_list_unkowns + obst_list_added + obst_list_unkowns_added

        current_target = target_routes[target_route_id]            
        route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy.idx, policy.Tr)
        route_update.compute_new_route()
        policy.Tr = route_update.new_track
        policy.idx = 0
        re_map_activation = 1

        # print("new tr ", policy.Tr)


    # COMPUTE: Policy               
    vr, vl = policy.step(state, vis=False)
    # print(vr, vl)
    # print(state)
    

    # Controller Output
    stop_now = int(stop_agent_flag == 0)
    vel_right = stop_now*brake_factor*max_vel*vr
    vel_left = stop_now*brake_factor*max_vel*vl
    policy_vels = [vel_right, vel_left]


    # Store
    state_storage.append( state )
    vel_storage.append( policy_vels )
    # print(vel_storage[i], vel_storage[i-1])
    # print(brake_factor)
    # print()


    # Stop Sim.
    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance        
    
    if (distance <= tol_goal) :            
        stop_agent_flag = 1
        break




# vis
# Results 
fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_storage]
state_y = [x[1] for x in state_storage]

trajectory_x = [val[0] for val in policy.Tr]
trajectory_y = [val[1] for val in policy.Tr]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 

ax.grid(True)
ax.axis('equal')
plt.show() 


# y = [3]
# y = y[2]

# Store Data
path = "./logging_data/data_saved/"
file_name = store_data.save_runData_pickle(agent_id, state_storage, selected_path, target_route_id, brake_actived, obs_detected, path)
print("File name = ", file_name)

#-----------------------------------------------------------------------------------------------------
#
#    Agent 2
#
# -------------------------------------- Controller -------------------------------------------------

# Model
# -------------------------------------- Agent 2
agent_id = 1

robot_prams['x_init'] = init_agent[agent_id][0]
robot_prams['y_init'] = init_agent[agent_id][1]
robot = model.model_diff_v1()
robot.initialize(robot_prams)
state = [robot.x, robot.y, robot.theta]


# Obstacle avoidance Tr.
obs_avoid = obs_avoidance.circle_avoidance()
obs_avoid.initialize(circle_avoidance_params)
# obs_avoid.angles = [180-a for a in range(15, 180, 15)]

# Obs. ALgorithm
obs_algorithm = obs_avoidance.obs_algorithm()
obs_algorithm_params['obs_method'] = obs_avoid
obs_algorithm.initialize(obs_algorithm_params, agent_id)

# Obstacle Detected
obs_detected = []

# Controller
# Keep in Mind: 
#   (*) dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]

# Prepare Trajectory for Controller
track = selected_path_a2
trajectory = [[track[0][i].item(), track[1][i].item()] for i in range(0, len(track[0]))]
pf_control_params['Tr'] = trajectory
policy = controllers.path_follower(pf_control_params)


# -------------------------------------- Map Update -------------------------------------------------
route_update = PSO_auxiliar.pso_map_update()
route_update.safe_margin_obs = safe_margin_obs_pso/divider_units
re_map_activation = 0

# -------------------------------------- Communication -------------------------------------------------
agent_id = 1
# Should Received Current Position Others
states_group = np.zeros((num_agents, 3))
states_group[0] = [init_agent[0][0], init_agent[0][1], 0]

# -------------------------------------- Simulaton -------------------------------------------------

# Store Vals.
# Dim : (sample_num, agent_idx, state[x,y,t])
state_storage = []
state_storage.append( state )

# Vr and Vl
vel_storage = [[0, 0]]

# Brake Plot Aux.
brake_actived = []

# Stop Signal
stop_agent_flag = 0

last_iteration_i = total_iter 
for i in range(0, total_iter):
    
    # print("Policy ", policy.Tr)
    vel_right = vel_storage[i][0]
    vel_left = vel_storage[i][1]

    # Applying control signal & Read Response
    robot.step(vel_right, vel_left)                                                     # model step
    state = [robot.x, robot.y, robot.theta]    

    # Read Swarm Position
    states_group[1] = state
    states_group[0] = [init_agent[0][0], init_agent[0][1], 0]    

    # Brake Gradually
    brake_factor, active_brake = proximity_action.limit_speed(state, states_group, agent_id, margin=brake_limit_margin)
    brake_actived.append( active_brake )

    # Obstacle detection
    obs_detection_flag = obs_avoidance.detect_obs_sim(state, obst_list_unkowns, detect_obs_params)
    obs_detected.append( obs_detection_flag )

    # Obstacle Avoidance
    if obs_detection_flag :
        # Emulate sensor
        sensor_dist = 25
    else:
        sensor_dist = 50

    # Collision Avoidance ALgorithm
    # obs_algorithm.check_sensor(sensor_dist, policy.idx, state, policy.Tr)
    # if obs_algorithm.controller_update :
    #     policy.Tr = obs_algorithm.Tr_obs            
    #     policy.idx = obs_algorithm.idx_output


    # Re-Map
    if remap_type == "Obstacle_detection" :
        remap_flag = obs_detection_flag
    elif remap_type == "new_data" :
        if time_map_update == i :
            remap_flag = True
        else:
            remap_flag = False

    if remap_flag and (re_map_activation==0):
        if remap_type == "Obstacle_detection" :
            new_obst_list = obst_list + obst_list_unkowns 
        elif remap_type == "new_data" :
            new_obst_list = obst_list + obst_list_unkowns + obst_list_added + obst_list_unkowns_added

        current_target = target_routes[target_route_id_a2]            
        route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy.idx, policy.Tr)
        route_update.compute_new_route()
        policy.Tr = route_update.new_track
        policy.idx = 0
        re_map_activation = 1


    # COMPUTE: Policy               
    vr, vl = policy.step(state, vis=False)

    # Controller Output
    stop_now = int(stop_agent_flag == 0)
    vel_right = stop_now*brake_factor*max_vel*vr
    vel_left = stop_now*brake_factor*max_vel*vl
    policy_vels = [vel_right, vel_left]


    # Store
    state_storage.append( state )
    vel_storage.append( policy_vels )


    # Stop Sim.
    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance        
    
    if (distance <= tol_goal) :            
        stop_agent_flag = 1
        break


# Store Data
path = "./logging_data/data_saved/"
file_name_a2 = store_data.save_runData_pickle(agent_id, state_storage, selected_path_a2, target_route_id, brake_actived, obs_detected, path)
print("File name = ", file_name)


# vis
# Results 
fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_storage]
state_y = [x[1] for x in state_storage]

trajectory_x = [val[0] for val in policy.Tr]
trajectory_y = [val[1] for val in policy.Tr]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 

ax.grid(True)
ax.axis('equal')
plt.show() 

# y = [0]
# y[2] = 2

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