#!/usr/bin/env python3

import rpyc
from ev3_utils import ev3_connection

import numpy as np
import math
import time

from PSO import pso_iterator
from PSO import PSO_decision
from PSO import PSO_auxiliar
from Aux_libs import ploting

from utils_fnc import op_funct, interpreter
from Env import env_small
from control import model, controllers, obs_avoidance, proximity_action

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.animation as animation

# ---------------------------------------------------------------------------------------------------
#                                           SETTINGS
# ---------------------------------------------------------------------------------------------------


total_iter = 50                             # For loop Iteretions (15000)
Ts = 0.4                                     # Sample time

max_vel = 1                                  # control*max
tol_goal = 0.1                                 # Stop signal when reach the goal [cm]


# Test Route
predefined_route_flag = False 


remap_flag = False                           # True : The route is re-computed
# remap_type = "Obstacle_detection" 
remap_type = "new_data"                      # Update base on new Obstacles discovered
remap_type = "None"
time_map_update = 1000                       # Iterationsto apply the remapping

# Small Environment
load_scene_params = {
    # 'scene_name' : 'scene_0_a5'
    # 'scene_name' : 'scene_0_a2'
    # 'scene_name' : 'scene_obs_0_a5'

    'scene_name' : 'scene_experiment_test0_a2'
    # 'scene_name' : 'scene_force_collision_obs'
    # 'scene_name' : 'scene_force_collision_obs_a1'
}

load_complementary_scene_params = {
    'scene_name' : 'scene_obs_0_a5_complement_1'
}

#
# EV3
#
# [ evdev_1, tickect ]
ip_list = ['10.16.12.26', '10.16.12.25']
ev3_params = {  
    'IP' : None,                            # WiFi IP
    'max_vel' : 200     # [deg/s]           # Limit (signal saturation)
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
    'resolution': 2
}

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
    'ray_length' : 25
}


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


# Agents
init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)
_, _, obst_list_added, obst_list_unkowns_added, _ = env_small.load_scene(load_complementary_scene_params)

# -------------------------------------- Trajectory Comp. ------------------------------------------------------


#
# PSO Computation
#
if predefined_route_flag == 0 : 
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


    init_agent, target_routes, obst_list, obst_list_unkowns, map_size, data_lists = interpreter.change_units(init_agent, target_routes, obst_list, obst_list_unkowns, map_size, data_lists, divider=100)
    ploting.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=True, obs_unknowns=obst_list_unkowns_converted, smaller=True)
else: 
    
    id_list = [i for i in range(0, len(init_agent))]    
    init_agent = [(0, 0), (0, 0.3)]
    target_routes = [
            [(1.0, 0)],                 # Route 1                 
            [(1.0, 0.3)] ]              # Route 2
    data_lists = [[[0, 0], [1.0, 0.0]],
                  [[0, 0.3], [1.0, 0.3]]]
    obst_original = ploting.convert_obs_coor(obst_list)
    obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)

    ploting.plot_scene(id_list, data_lists, obst_original, target_routes, cm_flag=True, obs_unknowns=obst_list_unkowns_converted, smaller=True)


# y= (0, 0)
# b = y[3] 
# -------------------------------------- EV3 Objects -------------------------------------------------

ev3_objects = []
for ip_i in ip_list:
    ev3_i = ev3_connection.ev3_object()
    ev3_params['IP'] = ip_i
    ev3_i.initialize(ev3_params)

    ev3_objects.append( ev3_i )


# -------------------------------------- Controller -------------------------------------------------

# Model
robot_list = []
state_list = []
for init in init_agent :
    robot_prams['x_init'] = init[0]
    robot_prams['y_init'] = init[1]
    robot = model.model_diff_v1()
    robot.initialize(robot_prams)
    state = [robot.x, robot.y, robot.theta]

    robot_list.append( robot )
    state_list.append( state )

# Total of robots in the Sim.
num_agents = len(robot_list)


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

# Controller
# Keep in Mind: 
#   (*) dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]

policy_list = []
for track in data_lists :
    # Prepare Trajectory for Controller
    if predefined_route_flag == 0 :
        trajectory = [[track[0][i].item(), track[1][i].item()] for i in range(0, len(track[0]))]
    pf_control_params['Tr'] = trajectory
    policy = controllers.path_follower(pf_control_params)

    # print()
    # print("TRACK = ", trajectory[0])

    policy_list.append( policy )


# -------------------------------------- Map Update -------------------------------------------------
route_update = PSO_auxiliar.pso_map_update()
route_update.safe_margin_obs = safe_margin_obs_pso
re_map_activation = [0 for i in range(0, num_agents)]

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

# For timing measure
first = [1 for i in range(0, num_agents)]
cycle_time = [0 for i in range(0, num_agents)] 

time_inside = np.zeros((total_iter+1, num_agents, 1))
cycle_time_storage = np.zeros((total_iter+1, num_agents, 1))

t_run_list = np.zeros((total_iter+1, num_agents, 1))
t_read_list = np.zeros((total_iter+1, num_agents, 1))


for ev3_i in ev3_objects:
    ev3_i.motors_stop()
    ev3_i.read_speed()    
    ev3_i.motors_run_forever(10, 10)


print("sim started ...")
last_iteration_i = total_iter 
for i in range(0, total_iter):
    
    for agent_idx, robot in enumerate(robot_list):
                
        # Receive State
        t_read_1 = time.time()
        vel_right, vel_left = ev3_objects[agent_idx].read_speed()
        t_read = time.time() - t_read_1
        print("Read Speed = ", vel_right, vel_left)
        vel_right, vel_left = op_funct.deg2m_two(vel_right, vel_left, robot.r_wheel)        # from deg/s to m/s
        if first[agent_idx] :
            Ts = 0
            first[agent_idx] = 0
        else:
            Ts = time.time() - cycle_time[agent_idx]                                        # compute sample time
            
        robot.Ts = Ts
        cycle_time[agent_idx] = time.time()        

        # Odometry
        robot.step(vel_right, vel_left)                                                     # model step
        state = [robot.x, robot.y, robot.theta]
        start_time = time.time()    

        # Policy Agent Idx
        policy_i = policy_list[agent_idx]


        # Brake Gradually
        brake_factor, active_brake = proximity_action.limit_speed(state, state_storage[i-1], agent_idx, margin=0.2)
        brake_actived[i, agent_idx, 0] = active_brake

        # Obstacle detection
        # sensor_dist_1 = ev3_objects[agent_idx].read_ultrasonic_in_m()
        sensor_dist_1 = 0.5
        obs_detection_flag = op_funct.is_smaller(sensor_dist_1, margin=0.25)
        obs_detected[i, agent_idx, 0] = obs_detection_flag
        
        # Collision Avoidance ALgorithm
        # obs_algorithm_i = obs_algorithm_avoidance[agent_idx]
        # obs_algorithm_i.check_sensor(sensor_dist, policy_i.idx, state, policy_i.Tr)        
        # if obs_algorithm_i.controller_update :
        #     policy_i.Tr = obs_algorithm_i.Tr_obs            
        #     policy_i.idx = obs_algorithm_i.idx_output


        # Re-Map
        if remap_type == "Obstacle_detection" :
            remap_flag = obs_detection_flag
        elif remap_type == "new_data" :
            if time_map_update == i :
                remap_flag = True
            else:
                remap_flag = False
        else:
           remap_flag = False


        if remap_flag and (re_map_activation[agent_idx]==0):
            if remap_type == "Obstacle_detection" :
                new_obst_list = obst_list + obst_list_unkowns 
            elif remap_type == "new_data" :
                new_obst_list = obst_list + obst_list_unkowns + obst_list_added + obst_list_unkowns_added

            current_target = target_routes[pso_routes.output_routes_ids[agent_idx]]            
            route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy_i.idx, policy_i.Tr)
            route_update.compute_new_route()
            policy_i.Tr = route_update.new_track
            policy_i.idx = 0
            re_map_activation[agent_idx] = 1


        # COMPUTE: Policy               
        vr, vl = policy_i.step(state, vis=False)

        # Controller Output
        stop_now = int(stop_agent_flag_list[agent_idx] == 0)
        vel_right = stop_now*brake_factor*max_vel*vr
        vel_left = stop_now*brake_factor*max_vel*vl
        policy_vels = [vel_right, vel_left]
        # print("VEls = ", policy_vels)


        # Storage
        state_i = np.array(state).reshape((1, 1, 3))
        state_storage[i+1, agent_idx, :] = state_i

        policy_vels_i = np.array(policy_vels).reshape((1, 1, 2))
        vel_storage[i+1, agent_idx, :] = policy_vels_i

        time_inside[i, agent_idx, 0] = time.time() - start_time
        cycle_time_storage[i, agent_idx, 0] = Ts

        # Send Policy Action 
        vr, vl = op_funct.m2deg_two(vel_right, vel_left, robot.r_wheel)
        vr = 200 
        vl = 200
        t_run_1 = time.time() 
        ev3_objects[agent_idx].motors_run_forever(vr, vl)
        t_run = time.time() - t_run_1

        
        t_run_list[i, agent_idx, 0] = t_run
        t_read_list[i, agent_idx, 0] = t_read

        # Stop Sim.
        distance = op_funct.distance(robot.x, robot.y, policy_i.Tr[-1][0], policy_i.Tr[-1][1])    # compute distance
        print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy_i.Tr[-1][0], policy_i.Tr[-1][1]) ) )
        print("Time Ts", agent_idx,  Ts)
        print("Run function time ", t_run)
        print()
        
        stop_agent_flag = stop_agent_flag_list[agent_idx]
        if (distance <= tol_goal) or (stop_agent_flag) :
            stop_list[agent_idx] = 1
 
            stop_agent_flag_list[agent_idx] = 1


    # Sim. Ends
    stop_flag = np.sum(stop_list)        
    if stop_flag >= num_agents :
        last_iteration_i = i        
        break
    
    
# Brake
for ev3_i in ev3_objects:
    ev3_i.motors_stop()

# Vis
print()
print("Last Position = ", state_storage[last_iteration_i,:,:])
print("Ts avarege (cycle time) = ", np.mean(cycle_time_storage[:last_iteration_i,:,:]), np.max(cycle_time_storage[:last_iteration_i,:,:]), np.min(cycle_time_storage[:last_iteration_i,:,:]))
print("Time inside avarege     = ", np.mean(time_inside[:last_iteration_i,:,:]))
print("Time read     = ", np.mean(t_read_list[:last_iteration_i,:,:]), np.max(t_read_list[:last_iteration_i,:,:]), np.min(t_read_list[:last_iteration_i,:,:]))
print("Time run     = ", np.mean(t_run_list[:last_iteration_i,:,:]), np.max(t_run_list[:last_iteration_i,:,:]), np.min(t_run_list[:last_iteration_i,:,:]))
print()


# Delete non used rows
state_storage = state_storage[:last_iteration_i,:,:]


ploting.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=True, states=state_storage, obs_unknowns=obst_list_unkowns_converted, smaller=True)

scene_axes, state_lines, figure, axes = ploting.animate_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=True, states=state_storage, obs_unknowns=obst_list_unkowns_converted, smaller=True)

jump = 50
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


# ani.save(filename="/media/scene_0_a5.mp4", writer="ffmpeg")