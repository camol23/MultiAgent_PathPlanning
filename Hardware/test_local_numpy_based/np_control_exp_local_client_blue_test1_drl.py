#!/usr/bin/env python3

'''
    (*) Remember update the MAC address for the bluetooth in messaging file 
'''

import time 
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, MoveDifferential, SpeedDPS
from ev3dev2.sensor import INPUT_2
from ev3dev2.sensor.lego import UltrasonicSensor
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sound import Sound

from ev3dev2.wheel import EV3EducationSetTire
from ev3dev2.wheel import EV3Tire



# messaging
from communication.messaging import *

import numpy as np
import math
import time

from PSO import pso_iterator
from PSO import PSO_decision
from PSO import PSO_auxiliar
# from Aux_libs import ploting

from utils_fnc import op_funct, interpreter
from Env import env_small
from control import model, controllers, obs_avoidance, proximity_action, kalman
from logging_data import store_data

from DRL2 import observations_module, manage_policy, coor_sys_transform, numpy_layers


#------ Input to choose the Map
print("Type => (1: Map_obst_test1) or (2 (Map_obst_test01) or (3 (Map_obst_drl) or (ELSE : Line)")
map_type_input = int(input())
if map_type_input == 1:
    map_type = 'scene_experiment_test0_a2'
    PSO_resolution = 5

elif map_type_input == 2:
    map_type = 'scene_experiment_test01_a2'
    PSO_resolution = 5

elif map_type_input == 3:
    map_type = 'scene_experiment_test01_a2_s20'
    PSO_resolution = 5

else:
    map_type = 'scene_0_a2_exp_line'
    PSO_resolution = 2


# map_type = 'scene_experiment_test0_a2'
# PSO_resolution = 5

print('map_type ', map_type)
print()


# 'scene_name' : 'scene_experiment_test0_a2'
    # 'scene_name' : 'scene_0_a2_exp_line'
# ---------------------------------------------------------------------------------------------------
#                                           SETTINGS
# ---------------------------------------------------------------------------------------------------
agent_id = 1                                # ticket

T_wait = 0                                  # time.sleep per cycle
cycle_time = 0                              # time to reach the model_step
Ts = 0                                      # Sample time

obst_collision_assistance_flag = False
remap_flag = False                           # True : The route is re-computed
remap_type = "Obstacle_detection" 
# remap_type = "new_data"                    # Update base on new Obstacles discovered
time_map_update = 1000                       # Iterationsto apply the remapping

total_iter = 500 #800                           # For loop Iteretions
# Ts = 0.4                                   # Sample time for Ssim

# max_speed_sim = op_funct.deg2m(200, 0.056/2)
# print("max_speed_sim ", max_speed_sim)
# max_vel = max_speed_sim #0.001
max_vel = 1
max_vel_drl = 0.8 #0.5 #0.8

divider_units = 100
tol_goal = 20/divider_units                                 # Stop signal when reach the goal [cm]


# Small Environment
load_scene_params = {
    # 'scene_name' : 'scene_0_a5'
    # 'scene_name' : 'scene_0_a2'

    # 'scene_name' : 'scene_experiment_test0_a2'
    # 'scene_name' : 'scene_0_a2_exp_line'
    'scene_name' : map_type

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
safe_margin_obs_pso = 10 # [cm]

pso_params = {
    'iterations': 100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': PSO_resolution
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
    'iterations': 20, #100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 50, #100,
    'resolution': PSO_resolution-1
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
    'kw' : 1, #5,                 # Angular Vel. Gain (sim = 5) works = 5
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


ekf_params = {
    'H_rows' : 4,
    'H_colmns' : 3,
    'X0' : [0, 0, 0],
    'P0_factor' : 10,
    'Q_sigma_list' : [36e-6, 36e-6, 6*36e-6],           # Var. Prossesing Noise
    'R_sigma_list' : [4*36e-6, 4*36e-6, 4*36e-6, (140e-6)/100000]   # Var. Measurement Noise
}


# DRL frame
frame_transform_params = {
    'frame_scale' : 0.02, #0.02,
    'frame_size' : 20,
    'circ_margin' : 0.02,
    'obst_r_frame' : 4,
    'tol_sugbgoal' : 0.10,

    'detection_distance' : 0.36  # It's update when it's intantiate
}

# Observations DRL
module_obs_params = {
    'scale_pos' : 10,
    'scale_obs' : 4
}


# Linit Speed (swarm)
brake_limit_margin = 30/divider_units#20/divider_units

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

# Agents
init_agent, target_routes, obst_list, obst_list_unkowns, map_size = env_small.load_scene(load_scene_params)
_, _, obst_list_added, obst_list_unkowns_added, _ = env_small.load_scene(load_complementary_scene_params)

# Take Obstacle list
obst_original = op_funct.convert_obs_coor(obst_list)
obst_list_unkowns_converted = op_funct.convert_obs_coor(obst_list_unkowns)

num_agents = len(init_agent)


# -------------------------------------- Trajectory Comp. ------------------------------------------------------

# -------------------------------------- Agent 2
#
# PSO Computation
#
agent_id = 1
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


# -------------------------------------- SHARE DATA 
# Each agent received a dist_list  
SERVER = "F0:45:DA:11:92:74"  # One
client = BluetoothMailboxClient()

# The server must be started before the client!
print("establishing connection...")
client.connect(SERVER)
print("connected!")

# Prepare the messages box
mbox = TextMailbox("partner_data", client)
state_mbox = TextMailbox("state_data", client)
mbox.wait()
print(mbox.read())              # Server: mbox.send("SERVER ready to receive")

# Send our Distance List (Response)
data_list_str = interpreter.turn_list_in_str(pso_iter_an.dist_list)
mbox.send(data_list_str)

# Server State
mbox.wait()
data_for_dist_list = mbox.read()
print("dist_list a2 = ", data_for_dist_list)      

# Interpreter function for received data
# data_list_a2 = pso_iter_a2.dist_list
data_list_a2 = interpreter.dist_list_interpreter(data_for_dist_list)
print("Data list from PSO shape ", data_list_a2)


# dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function 
dist_matrix = np.array(pso_iter_an.dist_list).reshape((len(pso_iter_an.dist_list), 1))     
dist_an = np.array(data_list_a2).reshape((len(data_list_a2), 1))    
dist_matrix = np.hstack((dist_an, dist_matrix))

# --------------------------------------
#
# Decision Making
#
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

# -------------------------------------- Agent 2 
agent_id = 1
target_route_id = traditional_path_route_id[agent_id]
selected_path = pso_iter_an.paths[target_route_id]

# Virtual Just to Unit. convertion
data_lists = [selected_path, selected_path]

# Units converted (REQUIRE)
init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, data_lists = interpreter.change_units(init_agent, target_routes, obst_original, obst_list_unkowns_converted, map_size, data_lists, divider=divider_units)
selected_path = data_lists[0] 
_, _, obst_list, obst_list_unkowns, _, _ = interpreter.change_units(init_agent, target_routes, obst_list, obst_list_unkowns, map_size, [], divider=divider_units)


# ------------------------------------------------------------------

# --------------------------- EV3 ----------------------------------

# Motors
motor_right = LargeMotor(OUTPUT_C)
motor_left = LargeMotor(OUTPUT_B)
# motor_right.reset()
# motor_left.reset()

ultrasonic_sensor = UltrasonicSensor(INPUT_2)

size_w = 87 
mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3Tire, size_w) # 87 with speed = 300
mdiff.gyro = GyroSensor()
mdiff.gyro.calibrate()      
mdiff.odometry_start(theta_degrees_start=0.0, x_pos_start=0.0, y_pos_start=0.0) # Not accurate

ev3_spk = Sound()

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
track = selected_path
# print(" POLICY")
# print(track)
trajectory = [[track[0][i].item(), track[1][i].item()] for i in range(0, len(track[0]))]
pf_control_params['Tr'] = trajectory
policy = controllers.path_follower(pf_control_params)


# EKF
ekf_params['X0'] = [init_agent[agent_id][0], init_agent[agent_id][1], 0]
ekf = kalman.kalman()
ekf.initialization(ekf_params)


# For DRL model in numpy
params_name = './logging_data/actor_all_18_np_params_list'
data_dict = store_data.load_time_data_pickle(params_name)
w_params_list = data_dict['w_params']
b_params_list = data_dict['b_params']

w_params = [np.array(data, dtype=np.float32) for data in w_params_list]
b_params = [np.array(data, dtype=np.float32) for data in b_params_list]
# DRL Agent
# actor_tf = actor_all_18_tf.load_model()
actor = numpy_layers.model_all_18()
actor.initialization(w_params, b_params)

# Obs. Module
obs_module = observations_module.module_observations()
obs_module.initialize(module_obs_params)
obs_module.WP0 = [0, 0]

trajectory = []
policy_drl = manage_policy.policy_manager()
policy_drl.initialization(0, trajectory, actor, obs_module)

# reframe object
frame_transform_params['detection_distance'] = 0.35
re_frame = coor_sys_transform.frame_transform()
re_frame.initialization(frame_transform_params)


# -------------------------------------- Map Update -------------------------------------------------
route_update = PSO_auxiliar.pso_map_update()
route_update.safe_margin_obs = safe_margin_obs_pso/divider_units
re_map_activation = 0


# -------------------------------------- Communication -------------------------------------------------
time.sleep(5)
print("Init Data send")
# Should Received Current Position from Others
communication_delay_counter_limit = 5
communication_delay_counter = 0

# Send our Init State
state_str = interpreter.turn_list_in_str(state)
state_mbox.send(state_str)

# Read Init State (Syncronization)
state_mbox.wait()
state_received = state_mbox.read()
state_partner = interpreter.only_state_interpreter(state_received)


states_group = np.zeros((num_agents, 3))
states_group[0] = state_partner
states_group[1] = state

# -------------------------------------- Running -------------------------------------------------

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

controller_time = []
actor_drl_time = []

first = 1
last_iteration_i = total_iter 
print("Running ...")
for i in range(0, total_iter):
    # Start time
    start_time = time.time()

    # Read Speed
    vel_right = motor_right.speed
    vel_left = motor_left.speed
    # print(vel_right, vel_left)
    vel_right, vel_left = op_funct.deg2m_two(vel_right, vel_left, robot.r_wheel)        # from deg/s to m/s
    if first :
        Ts = 0
        first = 0
    else:
        Ts = time.time() - cycle_time                                                       # compute sample time
        
    robot.Ts = Ts
    cycle_time = time.time()

    # Update Model (Odometry)
    robot.step(vel_right, vel_left)                                                     # model step
    state = [robot.x, robot.y, robot.theta]    
    vels = [robot.Vx, robot.Vy, robot.w] 

    # Read Swarm Position
    if communication_delay_counter >= communication_delay_counter_limit :
        state_received = state_mbox.read()
        state_partner = interpreter.only_state_interpreter(state_received)
        # states_group[1] = [init_agent[1][0], init_agent[1][1], 0]    
        states_group[0] = state_partner    
        states_group[1] = state

        # Send our Init State
        state_str = interpreter.turn_list_in_str(state)
        state_mbox.send(state_str)    
        
        communication_delay_counter = -1
    communication_delay_counter += 1

    # Brake Gradually
    brake_factor, active_brake = proximity_action.limit_speed(state, states_group, agent_id, margin=brake_limit_margin)
    brake_actived.append( active_brake )

    # Obstacle detection
    sensor_dist_1 = ultrasonic_sensor.distance_centimeters    
    sensor_dist_1 = sensor_dist_1/100
    dect_dist = 0.35 #0.25
    if sensor_dist_1 <= dect_dist and (policy.idx < (len(policy.Tr)-2)) and (active_brake==0):
        obs_detection_flag = True
        print("DEtect Obst. in state", state)
        
        # tone(frequency, duration)
        ev3_spk.tone(382, 10)
    else:
        obs_detection_flag = False

    obs_detected.append( obs_detection_flag )

    # Obstacle Avoidance
    # Collision Avoidance ALgorithm
    # obs_algorithm.check_sensor(sensor_dist_1, policy.idx, state[0:2], trajectory)
    # if obs_algorithm.controller_update :
    #     policy.Tr = obs_algorithm.Tr_obs
    #     policy.idx = 0


    # Re-Map
    if obst_collision_assistance_flag :
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

            motor_right.stop()
            motor_left.stop()

            current_target = target_routes[target_route_id]            
            route_update.initialization(map_size, state, current_target, pso_params_map_update, new_obst_list, policy.idx, policy.Tr)
            route_update.compute_new_route()
            policy.Tr = route_update.new_track
            policy.idx = 0
            re_map_activation = 1

            # print("new tr ", policy.Tr)


    goal_coor = policy.Tr[-2]
    
    # if obs_detection_flag and not(re_frame.first_detection):
    #     print("First detection sensor = ", sensor_dist_1)
    #     motor_right.stop()
    #     motor_left.stop()

    #     re_frame_stop_flag = True
    # else:
    #     re_frame_stop_flag = False

    re_frame.check_activation(state, obs_detection_flag, goal_coor)

    # COMPUTE: Policy               
    if  re_frame.its_on : 
        start = time.time()                 
        state_frame, sensor_dist_frame = re_frame.coor_transformation(state, sensor_dist_1)
        policy_drl.idx = 0
        policy_drl.Tr = [[state_frame[0], state_frame[1]], [re_frame.xg_sub_frame, re_frame.yg_sub_frame]]
        policy_drl.observations_module.WP0 = [state_frame[0], state_frame[1]]
        vr, vl = policy_drl.step_np(state_frame, vels, obst_flag=True, sensor_dist=sensor_dist_frame, obst_center_dist=module_obs_params['scale_obs'])            

        end_time = time.time() - start
        actor_drl_time.append(end_time)
        # print(state_frame, sensor_dist_frame)
        # if re_frame_stop_flag :
        #     max_vel_vr = 0
        #     max_vel_vl = 0
        # else:    
        #     max_vel_vr = max_vel_drl*vr  
        #     max_vel_vl = max_vel_drl*vl 

        vel_limit_ev3 = 200
        max_vel_vr = max_vel_drl*vr  
        max_vel_vl = max_vel_drl*vl 
    else:        
        start = time.time()
        vr, vl = policy.step(state, vis=False)    

        end_time = time.time() - start
        controller_time.append(end_time)
        # print(vr, vl)
        # print(state)
        max_vel_vr = max_vel*vr
        max_vel_vl = max_vel*vl

        vel_limit_ev3 = 200
    
    # Controller Output
    stop_now = int(stop_agent_flag == 0)
    vel_right = stop_now*brake_factor*max_vel_vr
    vel_left = stop_now*brake_factor*max_vel_vl
    # policy_vels = [vel_right, vel_left]

    vr, vl = op_funct.m2deg_two(vel_right, vel_left, robot.r_wheel)
    policy_vels = [vr, vl]

    # Store
    state_storage.append( state )
    vel_storage.append( policy_vels )
    # print(vel_storage[i], vel_storage[i-1])
    # print(brake_factor)
    # print()

    vr = max( min(vel_limit_ev3, float(vr) ), 0)
    vl = max( min(vel_limit_ev3, float(vl) ), 0)
    # mbox.send(str(vr)+","+str(vl)+","+str(stop))
    motor_right.speed_sp = vr
    motor_left.speed_sp = vl
    motor_right.run_forever()
    motor_left.run_forever() 

    # Stop Sim.
    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance        
    # print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy.Tr[-1][0], policy.Tr[-1][1]) ) )
    
    if (distance <= tol_goal) :            
        stop_agent_flag = 1
        print()
        print("STOP", robot.x, robot.y)
        print()
        break

    time.sleep(T_wait)
    end_time = time.time()

    # Vis.
    # print("*** Total time = ", end_time-start_time)

print()
print("End ", i)
print("Execution time 1-Iter. = ", end_time-start_time)
print("Final Position", robot.x, robot.y)
# Brake
motor_right.stop()
motor_left.stop()

print('controller_time ', np.mean(controller_time), np.var(controller_time), np.min(controller_time), np.max(controller_time))
print('actor_drl_time', np.mean(actor_drl_time), np.var(actor_drl_time), np.min(actor_drl_time), np.max(actor_drl_time))

with open("./logging_data/dist_test0.txt", "a") as f:
  f.write("\n")
  f.write("New test\n")
  f.write("Position = "+ str(robot.x) + " " + str(robot.y) + " "+ str(robot.theta) +"\n")


# Store Data
path = "./logging_data/data_saved/"
file_name = store_data.save_runData_pickle(agent_id, state_storage, selected_path, target_route_id, brake_actived, obs_detected, path)
print("File name = ", file_name)


# ----- END
print("Waiting for Others to End")

flag = True
while(flag):
    data = mbox.read()
    # print("data received = ", data)
    if data == "Ok":
        flag = False

    time.sleep(0.5)
    mbox.send("Ok")

print("Finally End")
