import numpy as np
import math
import random
import time

from PSO import pso_iterator
from PSO import PSO_decision
from PSO import PSO_auxiliar
from Aux_libs import ploting

from utils_fnc import op_funct, interpreter, opt_funct_numpy
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

save_time = False
remap_type = "Obstacle_detection" 
# remap_type = "new_data"                      # Update base on new Obstacles discovered
time_map_update = 1000                       # Iterationsto apply the remapping

total_iter = 15000                           # For loop Iteretions
Ts = 0.4                                     # Sample time

max_vel = 1
tol_goal = 1                                 # Stop signal when reach the goal [cm]

# Add sensor noise
flag_noise_sensor = True
mu = 0
sigma = 0.5 # 5mm 
# sensor_noise = random.gauss(mu, sigma)

# Add noise to the Obst.
noise_obst_flag = True
sigma_obst = 1 # 1cm 

# Remap PSO
remap_persistance_flag = 0          # Activates the Re-mapping with PSO in persistance Mode
                                    # 0: One-Time   
                                    # 1:Share information     
                                    # 2: No shared information
                                    # 3 >= ReMap based on Obst. ID 
obs_algorithm_avoidance_flag = False
# map
map_size = (250, 250)

# Just a Variable
remap_flag = False                           # True : The route is re-computed

# Small Environment
load_scene_params = {
    # 'scene_name' : 'scene_0_a5'
    # 'scene_name' : 'scene_0_a2'
    # 'scene_name' : 'scene_experiment_test0_a2'

    # 'scene_name' : 'scene_experiment_test1_a2'
    'scene_name' : 'scene_experiment_test01_a2'
    
    # 'scene_name' : 'scene_obs_0_a5'        
    # 'scene_name' : 'scene_obs_2_a5'       #
    # 'scene_name' : 'scene_obs_3_a5'       #     
    # 'scene_name' : 'scene_2_obs_5_a5'     #  
    # 'scene_name' : 'scene_2_obs_7_a5'     #

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
safe_margin_obs_pso = 10 # 

pso_params = {
    'iterations': 100,  #100 # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': 5 #12 # 5
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
    'iterations': 40, #120, #exp40,  #100 # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 50, #100,
    'resolution': 4 #6 #exp4 #6# 5
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
    'ray_length' : 30 #25
}


obst_persistance_params = {
    'counter_limit' : 40,          # 500 with /4 (stable)
    'units_divider' : 1,            # 1 for cm
    'default_obst_width' : 30, #20,      # cm
    'default_obst_height' : 30, #40,
    'detection_margin' : 25           # cm
}

# ---------------------------------------------------------------------------------------------------
# -----------------------------------------------Load Scene ---------------------------------------------


# Agents
init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)
_, _, obst_list_added, obst_list_unkowns_added, _ = env_small.load_scene(load_complementary_scene_params)

map_density = opt_funct_numpy.density_map(map_size, obst_list, obst_list_unkowns)
num_obst_knowns = len(obst_list)
num_obst_unknowns = len(obst_list_unkowns)
num_obst = num_obst_knowns + num_obst_unknowns

# Save the Obst. without noise (Plotting porpuses)
obst_original = ploting.convert_obs_coor(obst_list)
obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)

# Add Noise to the Obst.
if noise_obst_flag :
    obst_list, obst_list_unkowns = interpreter.add_obst_noise(obst_list, obst_list_unkowns, mu = 0, sigma = sigma_obst)


# -------------------------------------- Trajectory Comp. ------------------------------------------------------


#
# PSO Computation
#
dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function
an_possible_paths = []                                      # All the computed paths
data_lists = []                                             # Selected Agent Paths

pso_routes_time = np.zeros((len(init_agent), 1))
# Agent n
for i in range(0, len(init_agent)):

    start_time = time.time()
    pso_iter_an = pso_iterator.PSO_wrapper()
    pso_iter_an.initialization(map_size, init_agent[i], target_routes, pso_params, obst_list)
    pso_iter_an.safe_margin_obs = safe_margin_obs_pso
    pso_iter_an.itereration(plot_flag=Plot_from_PSO)
    
    end_time = time.time() - start_time
    pso_routes_time[i, 0] = end_time

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
print(dist_matrix.shape)

# Take Obstacle list
# obst_original = pso_iter_an.obs_rect_list_original
# obst_list_unkowns_converted = ploting.convert_obs_coor(obst_list_unkowns)

#
# Decision Making
#
# Test PSO for Assigning routes task
start_time = time.time()
pso_routes = PSO_decision.PSO()
pso_routes.initialization(pso_params_routes, dist_matrix)
pso_routes.pso_compute()

pso_decision_time = time.time() - start_time
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


ploting.plot_scene(pso_routes.output_routes_ids, data_lists, obst_original, target_routes, cm_flag=True, obs_unknowns=obst_list_unkowns_converted)



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

persistance_pso_aux = PSO_auxiliar.obst_persistance()
persistance_pso_aux.initialization(obst_persistance_params)

# Cooldown Method (Verify to Remap each dt)
cooldown_remap =  18 #25
cooldown_remap_counter = [0 for i in range(0, num_agents)]
print("Cooldown Remap Limit ", cooldown_remap)

# Obs. Persistance (Count the consecutive Obst. Detections)
# No share Information
persistance_pso_aux_list = []
for i in range(0, num_agents):
    persistance_pso_aux = PSO_auxiliar.obst_persistance()
    persistance_pso_aux.initialization(obst_persistance_params)

    persistance_pso_aux_list.append(persistance_pso_aux)

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

sensor_dist_list = np.zeros((total_iter+1, num_agents, 1))

# Discover Unkonws
obst_list_discover_unkowns = []
obst_ids_discovered = []

# Times
execution_time = np.zeros((total_iter+1, num_agents, 1))
remapping_time = np.zeros((total_iter+1, num_agents, 1))
avoidace_time = np.zeros((total_iter+1, num_agents, 1))
policy_time = np.zeros((total_iter+1, num_agents, 1))

last_iteration_i = total_iter 
for i in range(0, total_iter):
    
    for agent_idx, robot in enumerate(robot_list):
        start_exe_time = time.time()

        vel_right = vel_storage[i, agent_idx, 0]
        vel_left = vel_storage[i, agent_idx, 1]

        # Applying control signal & Read Response
        robot.step(vel_right, vel_left)                                                     # model step
        state = [robot.x, robot.y, robot.theta]    

        # Policy Agent Idx
        policy_i = policy_list[agent_idx]

        # Brake Gradually
        brake_factor, active_brake = proximity_action.limit_speed(state, state_storage[i-1], agent_idx, margin=20)
        brake_actived[i, agent_idx, 0] = active_brake

        # Obstacle detection
        obs_detection_flag, sensor_dist, obst_id = obs_avoidance.detect_obs_sim_v2(state, obst_list_unkowns, detect_obs_params)
        obs_detected[i, agent_idx, 0] = obs_detection_flag

        # Obstacle Avoidance        
        if flag_noise_sensor:
            sensor_noise = random.gauss(mu, sigma)
            sensor_dist = sensor_dist + sensor_noise

        sensor_dist_list[i, agent_idx, 0] = sensor_dist
        if obs_detection_flag :
            # Emulate sensor
            # print("Sensor dist. ", sensor_dist)
            # sensor_dist = 25            
            pass
        else:
            sensor_dist = 50

        # Collision Avoidance ALgorithm
        if obs_algorithm_avoidance_flag :
            start_time = time.time()

            obs_algorithm_i = obs_algorithm_avoidance[agent_idx]
            obs_algorithm_i.check_sensor(sensor_dist, policy_i.idx, state, policy_i.Tr)
            if obs_algorithm_i.controller_update :
                policy_i.Tr = obs_algorithm_i.Tr_obs            
                policy_i.idx = obs_algorithm_i.idx_output
            
            avoidace_time[i, agent_idx, 0] = time.time() - start_time
        else:
        #------------------------------------------------------------- Re-Map  Modes -------------------------------------------------------------------------------------------------- 
            start_time_remapping = time.time()

            if remap_type == "Obstacle_detection" :
                remap_flag = obs_detection_flag
            elif remap_type == "new_data" :
                if time_map_update == i :
                    remap_flag = True
                else:
                    remap_flag = False

            if remap_persistance_flag == 0 : 
                # One Time detection
                if remap_flag and (re_map_activation[agent_idx]==0):
                    if remap_type == "Obstacle_detection" :
                        # new_obst_list = obst_list + obst_list_unkowns
                        already_discovered = False
                        for id_discoverd in obst_ids_discovered:
                            if id_discoverd == obst_id :
                                already_discovered = True
                                break

                        if (obst_id != -1) and (already_discovered==0) :
                            obst_list_discover_unkowns.append(obst_list_unkowns[obst_id])
                            new_obst_list = obst_list + obst_list_discover_unkowns
                        else:
                            new_obst_list = obst_list + obst_list_discover_unkowns
                    elif remap_type == "new_data" :
                        new_obst_list = obst_list + obst_list_unkowns + obst_list_added + obst_list_unkowns_added

                    current_target = target_routes[pso_routes.output_routes_ids[agent_idx]]            
                    route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy_i.idx, policy_i.Tr)
                    route_update.compute_new_route()
                    policy_i.Tr = route_update.new_track
                    policy_i.idx = 0

                    re_map_activation[agent_idx] = 1
            
            elif remap_persistance_flag == 1 : 
                # All share information about Obst. found during running
                if remap_flag:
                                    
                    if persistance_pso_aux.activation_flag :
                        # Include Unkown Obst. guess location
                        # print("COUTER ", persistance_pso_aux.counter)
                        persistance_pso_aux.add_obst_by_detection(state, sensor_dist)                    
                        # Update Map
                        obst_list_unkowns_found = persistance_pso_aux.unknown_obst_list
                        new_obst_list = obst_list + obst_list_unkowns_found + obst_list_added + obst_list_unkowns_added

                        # Re-compute Trajectory (PSO)
                        current_target = target_routes[pso_routes.output_routes_ids[agent_idx]]            
                        route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy_i.idx, policy_i.Tr)
                        route_update.compute_new_route()
                        policy_i.Tr = route_update.new_track
                        policy_i.idx = 0

                        new_obst_list = []
                    
                    persistance_pso_aux.detection_counter()

            elif remap_persistance_flag == 2 :
                # No information is shared            
                if remap_flag:
                    
                    persistance_pso_aux_i = persistance_pso_aux_list[agent_idx]
                    if persistance_pso_aux_i.activation_flag :
                        # Include Unkown Obst. guess location
                        persistance_pso_aux_i.add_obst_by_detection(state, sensor_dist)                    
                        # Update Map
                        obst_list_unkowns_found = persistance_pso_aux_i.unknown_obst_list
                        new_obst_list = obst_list + obst_list_unkowns_found + obst_list_added + obst_list_unkowns_added

                        # Re-compute Trajectory (PSO)
                        current_target = target_routes[pso_routes.output_routes_ids[agent_idx]]            
                        route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy_i.idx, policy_i.Tr)
                        route_update.compute_new_route()
                        policy_i.Tr = route_update.new_track
                        policy_i.idx = 0

                        new_obst_list = []
                    
                    persistance_pso_aux_i.detection_counter()
                    persistance_pso_aux_list[agent_idx] = persistance_pso_aux_i

            else:            
                # Re-Map based on Obstacle ID (semi-Unnkown)
                if cooldown_remap_counter[agent_idx] >= cooldown_remap :
                    cooldown_remap_counter[agent_idx] = 0
                    if remap_flag :
                        # new_obst_list = obst_list + obst_list_unkowns
                        if obst_id != -1 :
                                obst_list_discover_unkowns.append(obst_list_unkowns[obst_id])
                                new_obst_list = obst_list + obst_list_discover_unkowns
                        else:
                            new_obst_list = obst_list + obst_list_discover_unkowns
                        

                        current_target = target_routes[pso_routes.output_routes_ids[agent_idx]]            
                        route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy_i.idx, policy_i.Tr)
                        route_update.compute_new_route()
                        policy_i.Tr = route_update.new_track
                        policy_i.idx = 0
            
            remapping_time[i, agent_idx, 0] = time.time() - start_time_remapping               

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # COMPUTE: Policy       
        start_time = time.time()
        vr, vl = policy_i.step(state, vis=False)

        policy_time[i, agent_idx, 0] = time.time() - start_time

        # Controller Output
        stop_now = int(stop_agent_flag_list[agent_idx] == 0)
        vel_right = stop_now*brake_factor*max_vel*vr
        vel_left = stop_now*brake_factor*max_vel*vl
        policy_vels = [vel_right, vel_left]
        # print("VEls = ", policy_vels)

        # Store
        state_i = np.array(state).reshape((1, 1, 3))
        state_storage[i+1, agent_idx, :] = state_i

        policy_vels_i = np.array(policy_vels).reshape((1, 1, 2))
        vel_storage[i+1, agent_idx, :] = policy_vels_i


        # Stop Sim.
        distance = op_funct.distance(robot.x, robot.y, policy_i.Tr[-1][0], policy_i.Tr[-1][1])    # compute distance
        # print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy_i.Tr[-1][0], policy_i.Tr[-1][1]) ) )
        
        stop_agent_flag = stop_agent_flag_list[agent_idx]
        if (distance <= tol_goal) or (stop_agent_flag) :
            stop_list[agent_idx] = 1
            # if agent_idx == 2:
                # print("STOP Agent = ", agent_idx,  robot.x, robot.y, distance)
                # print()

            # Turn-off agent
            #vel_storage[i+1, agent_idx, :] = np.zeros((1, 1, 2))
            # vel_storage[i+1, agent_idx, 0] = 0
            # vel_storage[i+1, agent_idx, 1] = 0
            stop_agent_flag_list[agent_idx] = 1
        
        # Time 
        execution_time[i, agent_idx, 0] = time.time() - start_exe_time

    # Draw Scene
    # ploter.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, state_storage)

    # Counter for Test Re-Map
    for a in range(0, num_agents):
        current_val = cooldown_remap_counter[a]
        if current_val > 35000 :
            current_val = 0
        cooldown_remap_counter[a] = current_val + 1

    # Sim. Ends
    stop_flag = np.sum(stop_list)        
    if stop_flag >= num_agents :
        last_iteration_i = i        
        break
    


# Delete non used rows
state_storage = state_storage[:last_iteration_i,:,:]

# Time measurements
execution_time = execution_time[:last_iteration_i, :, :]
avoidace_time = avoidace_time[:last_iteration_i, :, :]
remapping_time = remapping_time[:last_iteration_i, :, :]
policy_time = policy_time[:last_iteration_i, :, :]

print()
print("Simulation Data for ", load_scene_params['scene_name'])
print("Number of Obstacles {} | # knowns {} | # Unkowns {}".format(num_obst, num_obst_knowns, num_obst_unknowns))
print("execution_time mean",  np.mean(execution_time, axis=0) )
print("execution_time std",  np.std(execution_time, axis=0) )
print("execution_time mean total",  np.mean(execution_time) )
print("execution_time std total",  np.std(execution_time) )
print("avoidace_time",  np.mean(avoidace_time, axis=0) )
print("remapping_time",  np.mean(remapping_time, axis=0) )
print("policy_time",  np.mean(policy_time, axis=0) )
print("pso_routes_time",  np.mean(pso_routes_time, axis=0) )
print("pso_decision_time",  pso_decision_time )
print()
total_distance = opt_funct_numpy.data_state_distance(state_storage)
print("Total distance = ", total_distance)
print("Map Density ", map_density)
print()


# Save time
if save_time :
    if obs_algorithm_avoidance_flag :
        remap_persistance_flag = 4 #(Obstacle avoidance algorithm)

    time_dict = {
        "type" : remap_persistance_flag,
        "execution_time" : execution_time,
        "avoidace_time" : avoidace_time,
        "remapping_time" : remapping_time,
        "policy_time" : policy_time, 
        "pso_routes_time" : pso_routes_time, 
        "pso_decision_time" :  pso_decision_time,
        "total_distance" : total_distance,
        "map_density" : map_density,
        "num_obst" : num_obst, 
        "num_obst_knowns" : num_obst_knowns,
        "num_obst_unknowns" : num_obst_unknowns
    }
    

    file_name = load_scene_params['scene_name'] +"_method_"+ str(remap_persistance_flag)
    file_path = "./logging_data/time_test/"
    store_data.save_time_data_pickle(file_name, time_dict, file_path)




# Plotting Trajectories
ploting.plot_scene(pso_routes.output_routes_ids, data_lists, obst_original, target_routes, cm_flag=True, states=state_storage, obs_unknowns=obst_list_unkowns_converted)


scene_axes, state_lines, figure, axes = ploting.animate_scene(pso_routes.output_routes_ids, data_lists, obst_original, target_routes, cm_flag=True, states=state_storage, obs_unknowns=obst_list_unkowns_converted)

# state_axes = [axes.plot([], []) for i in range(0, num_agents)]
state_axes = []
for i in range(0, num_agents):
    pplot_ax, = axes.plot([], [])
    state_axes.append(pplot_ax)

jump = 50
def animate_update(frame):
    
    state_lines_2 = []
    text_list = []
    for i in range(0, num_agents):

        if brake_actived[frame*jump, i, 0] :
            color_agent = 'black'

        elif obs_detected[frame*jump, i, 0] :
            color_agent = 'tomato'

        else:
            # color_agent = 'blue'
            color_agent = mcolors.CSS4_COLORS[ploting.colors_agent[i]]
    
        # state_lines_2.append( axes.plot(state_storage[:frame*jump, i, 0], state_storage[:frame*jump, i, 1], color= mcolors.CSS4_COLORS[color_agent], ls='--') )
        # state_lines_2.append( axes.plot(state_storage[:frame*jump, i, 0], state_storage[:frame*jump, i, 1], color=color_agent, ls='--') )

        state_axes[i].set_data(state_storage[:frame*jump, i, 0], state_storage[:frame*jump, i, 1])
        state_axes[i].set_color(color_agent)
        state_axes[i].set_linestyle('--')


        # text_x = [(x*30)+10 for x in range(0, num_agents)]
        # text_y = [10 for y in range(0, num_agents)]
        # text_list.append( axes.text(text_x[i], text_y[i], str(sensor_dist_list[frame*jump, i, 0]), fontsize=12, horizontalalignment='left',
        #             verticalalignment='center', color=mcolors.CSS4_COLORS['black']) )
            

    # return scene_axes + state_lines_2 + text_list
    return scene_axes + state_axes + text_list


ani = animation.FuncAnimation(fig=figure, func=animate_update, frames=int(total_iter/jump), interval=Ts*1000)
# plt.tight_layout()
plt.show()


# ani.save(filename="/media/scene_0_a5.mp4", writer="ffmpeg")

print()
print("NO LABELS")
print(load_scene_params['scene_name'] + "_method_"+ str(remap_persistance_flag))
print(str(num_obst), str(num_obst_knowns), str(num_obst_unknowns))
print(np.mean(execution_time, axis=0) )
print(np.mean(avoidace_time, axis=0) )
print(np.mean(remapping_time, axis=0) )
print(np.mean(policy_time, axis=0) )
print(np.mean(pso_routes_time, axis=0) )
print(pso_decision_time )
print()
total_distance = opt_funct_numpy.data_state_distance(state_storage)
print(total_distance)
print(map_density)
print()