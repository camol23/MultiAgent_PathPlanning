import tensorflow as tf
import numpy as np
import math
import copy
import time

from PSO import pso_iterator
from PSO import PSO_decision
from Aux_libs import ploting

from DRL2 import observations_module, manage_policy
import actor_tg4_tf

from utils_fnc import op_funct
from Env import env_small
from control import model, controllers, obs_avoidance, proximity_action

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.animation as animation

# ---------------------------------------------------------------------------------------------------
#                                           SETTINGS
# ---------------------------------------------------------------------------------------------------

total_iter = 300                           # For loop Iteretions

Ts = 0.1                                     # Sample time

max_vel = 2
tol_goal = 3                                 # Stop signal when reach the goal [cm]


# Small Environment
load_scene_params = {
    # 'scene_name' : 'scene_0_a5'
    # 'scene_name' : 'scene_0_a2'
    # 'scene_name' : 'scene_obs_0_a5'
    'scene_name' : 'scene_no_obst'
    # 'scene_name' : 'scene_force_collision_obs'
    # 'scene_name' : 'scene_force_collision_obs_a1'
}

# DRL
# Observations
module_obs_params = {
    'scale_pos' : 10,
    'scale_obs' : 4
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
    'resolution': 3
}

# PSO Decision Making
pso_params_routes = {
    'iterations': 20, 
    'w': 0.04, # 0.04
    'Cp': 0.1, #0.2, # 0.7
    'Cg': 0.6, # 0.1 # 0.1
    'num_particles': 30
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
    'ray_length' : 25
}


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


# Agents
# Obstacles
# x_botton, y_botton, width, height

init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)

# -------------------------------------- Trajectory Comp. ------------------------------------------------------


#
# PSO Computation
#
dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function
an_possible_paths = []                                      # All the computed paths
data_lists = []                                             # Selected Agent Paths

# Agent n
for i in range(0, len(init_agent)):

    pso_iter_an = pso_iterator.PSO_wrapper()
    pso_iter_an.initialization(map_size, init_agent[i], target_routes, pso_params, obst_list)
    pso_iter_an.safe_margin_obs = safe_margin_obs_pso
    pso_iter_an.itereration(plot_flag=Plot_from_PSO)

    print("agent ", i, " Dist. ", pso_iter_an.dist_list)
    print()

    # Stack Routes
    dist_an = np.array(pso_iter_an.dist_list).reshape((len(pso_iter_an.dist_list), 1))    
    dist_matrix = np.hstack((dist_matrix, dist_an))

    # Store possible Paths
    an_possible_paths.append( pso_iter_an.paths )



dist_matrix = np.delete(dist_matrix, 0, axis=1)
print("dist_matrix")
print(dist_matrix)

# Take Obstacle list
obst_original = pso_iter_an.obs_rect_list_original
obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)

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


# Take the chosen routes
for i, paths in enumerate(an_possible_paths) :
    selected_path = paths[pso_routes.output_routes_ids[i]]

    data_lists.append( selected_path )


ploting.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=False, obs_unknowns=obst_list_unkowns_converted)


# -------------------------------------- Map Units ------------------------------------------------- 

init_agent = [[val[0]/10, val[1]/10] for val in init_agent]
obst_list = [[val[0]/10, val[1]/10, val[2]/10, val[3]/10] for val in obst_list]
obst_list_unkowns = [[val[0]/10, val[1]/10, val[2]/10, val[3]/10] for val in obst_list_unkowns]
map_size = [map_size[0]/10, map_size[1]/10]

new_target = []
for target in target_routes:
    new_target.append( [(val[0]/10, val[1]/10) for val in target] )

print()
print("Target Routes")
print(target_routes)
target_routes = copy.deepcopy(new_target)
print(target_routes)
print()

route_x = []
route_y = []
new_datalist = []
print("data update")

    
for route in data_lists :
    new_datalist.append( route/10 )

print(data_lists)
print()
data_lists = copy.deepcopy(new_datalist)
print(data_lists)


# y= (0, 0)
# b = y[3] 

# -------------------------------------- Controller -------------------------------------------------

# Model
robot_list = []
state_list = []
vels_list = []
for init in init_agent :
    robot_prams['x_init'] = init[0]
    robot_prams['y_init'] = init[1]
    robot = model.model_diff_v1()
    robot.initialize(robot_prams)
    state = [robot.x, robot.y, robot.theta]
    vels = [robot.Vx, robot.Vy, robot.w]

    robot_list.append( robot )
    state_list.append( state )
    vels_list.append( vels )

# Total of robots in the Sim.
num_agents = len(robot_list)

# Controller (DRL)
# Keep in Mind: 
#   (*) dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]

policy_list = []
for i, track in enumerate(data_lists) :
    
    # DRL Agent
    actor = actor_tg4_tf.load_model()
    
    # Obs. Module
    obs_module = observations_module.module_observations()
    obs_module.initialize(module_obs_params)
    obs_module.WP0 = [init_agent[i][0], init_agent[i][1]]

    trajectory = [[track[0][i].item(), track[1][i].item()] for i in range(0, len(track[0]))]
    policy = manage_policy.policy_manager()
    policy.initialization(i, trajectory, actor, obs_module)

    policy_list.append( policy )


# -------------------------------------- Collision Avoidance -------------------------------------------------

obs_algorithm_avoidance = []
for i in range(0, num_agents):
    # Obstacle avoidance Tr.
    obs_avoid = obs_avoidance.circle_avoidance()
    obs_avoid.initialize(circle_avoidance_params)
    # obs_avoid.angles = [180-a for a in range(15, 180, 15)]

    # Obs. ALgorithm
    obs_algorithm = obs_avoidance.obs_algorithm()
    obs_algorithm_params['obs_method'] = obs_avoid
    obs_algorithm.initialize(obs_algorithm_params, i)

    obs_algorithm_avoidance.append( obs_algorithm )

# Obstacle Detected
obs_detected = np.zeros((total_iter+1, num_agents, 1))


# -------------------------------------- Simulaton -------------------------------------------------


# Store Vals.
# Dim : (sample_num, agent_idx, state[x,y,t])
state_storage = np.zeros((total_iter+1, num_agents, 3))
for i, state in enumerate(state_list) :
    state_i = np.array(state).reshape((1, 1, 3))
    state_storage[0, i, :] = state_i

# Vr and Vl
vel_storage = np.zeros((total_iter+1, num_agents, 2))

# Stop Signal
stop_list = np.zeros((num_agents, 1))

# Brake Plot Aux.
brake_actived = np.zeros((total_iter+1, num_agents, 1))

# Stop Signal
stop_agent_flag_list = [0 for i in range(0, len(robot_list))]


# Time 
policy_time = np.zeros((total_iter+1, num_agents, 1))

last_iteration_i = total_iter 
for i in range(0, total_iter):
    
    for agent_idx, robot in enumerate(robot_list):
        vel_right = vel_storage[i, agent_idx, 0]
        vel_left = vel_storage[i, agent_idx, 1]

        # Applying control signal & Read Response
        robot.step(vel_right, vel_left)                                                     # model step
        state = [robot.x, robot.y, robot.theta]   
        vels = [robot.Vx, robot.Vy, robot.w] 

        # Policy Agent Idx
        policy_i = policy_list[agent_idx]

        # Brake Gradually
        brake_factor, active_brake = proximity_action.limit_speed(state, state_storage[i-1], agent_idx, margin=20)
        brake_actived[i, agent_idx, 0] = active_brake

        # Obstacle detection
        # obs_detection_flag = obs_avoidance.detect_obs_sim(state, obst_list_unkowns, detect_obs_params)
        # obs_detected[i, agent_idx, 0] = obs_detection_flag

        # # Obstacle Avoidance
        # if obs_detection_flag :
        #     # Emulate sensor
        #     sensor_dist = 25
        # else:
        #     sensor_dist = 50

        # obs_algorithm_i = obs_algorithm_avoidance[agent_idx]
        # obs_algorithm_i.check_sensor(sensor_dist, policy_i.idx, state, policy_i.Tr)
        # if obs_algorithm_i.controller_update :
        #     policy_i.Tr = obs_algorithm_i.Tr_obs            
        #     policy_i.idx = obs_algorithm_i.idx_output
                

        # COMPUTE: Policy               
        start_time = time.time()
        vr, vl = policy_i.step(state, vels)

        policy_time[i, agent_idx, 0] = time.time() - start_time

        # Controller Output
        stop_now = int(stop_agent_flag_list[agent_idx] == 0)
        vel_right = stop_now*brake_factor*max_vel*vr
        vel_left = stop_now*brake_factor*max_vel*vl
        policy_vels = [vel_right, vel_left]


        # Store
        state_i = np.array(state).reshape((1, 1, 3))
        state_storage[i+1, agent_idx, :] = state_i

        policy_vels_i = np.array(policy_vels).reshape((1, 1, 2))
        vel_storage[i+1, agent_idx, :] = policy_vels_i


        # Stop Sim.
        distance = op_funct.distance(robot.x, robot.y, policy_i.Tr[-1][0], policy_i.Tr[-1][1])    # compute distance
        
        
        stop_agent_flag = stop_agent_flag_list[agent_idx]
        if (distance <= tol_goal) or (stop_agent_flag) :
            
            if stop_list[agent_idx] == 0:
                print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy_i.Tr[-1][0], policy_i.Tr[-1][1]) ) )                
            
            stop_list[agent_idx] = 1
            # if agent_idx == 2:
                # print("STOP Agent = ", agent_idx,  robot.x, robot.y, distance)
                # print()

            # Turn-off agent
            #vel_storage[i+1, agent_idx, :] = np.zeros((1, 1, 2))
            # vel_storage[i+1, agent_idx, 0] = 0
            # vel_storage[i+1, agent_idx, 1] = 0
            stop_agent_flag_list[agent_idx] = 1


    # Sim. Ends
    stop_flag = np.sum(stop_list)        
    if stop_flag >= num_agents :
        last_iteration_i = i        
        break
    
    


# -------------------------------------- Time -------------------------------------------------
policy_time = policy_time[:last_iteration_i, :, :]
print("policy_time",  np.mean(policy_time, axis=0) )
print("policy_time total mean",  np.mean(policy_time) )
print("policy_time total STD",  np.std(policy_time) )

# -------------------------------------- Visualization -------------------------------------------------

# Delete non used rows
state_storage = state_storage[:last_iteration_i,:,:]


ploting.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=False, states=state_storage, obs_unknowns=obst_list_unkowns_converted)

scene_axes, state_lines, figure, axes = ploting.animate_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=False, states=state_storage, obs_unknowns=obst_list_unkowns_converted)

jump = 5
def animate_update(frame):
    state_lines_2 = []
    
    for i in range(0, num_agents):
        
        if brake_actived[frame*jump, i, 0] :            
            color_agent = 'black'

        elif obs_detected[frame*jump, i, 0] :
            color_agent = 'tomato'

        else:
            color_agent = 'blue'
    
        state_lines_2.append( axes.plot(state_storage[:frame*jump, i, 0], state_storage[:frame*jump, i, 1], color= mcolors.CSS4_COLORS[color_agent]) )
            

    return scene_axes + state_lines_2


ani = animation.FuncAnimation(fig=figure, func=animate_update, frames=int(total_iter/jump), interval=Ts*1000)
# plt.tight_layout()
plt.show()

