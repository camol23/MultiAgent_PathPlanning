#!/usr/bin/env python3

import time 
from messaging import *

from utils_fnc import op_funct, interpreter
from control import model, controllers, obs_avoidance

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------
T_wait = 0                                  # time.sleep per cycle
cycle_time = 0                              # time to reach the model_step
Ts = 0                                      # Sample time

max_vel = 1
tol_goal = 0.02                            # Stop signal when reach the goal [m]

# Trajectory
x_init = 0.25
y_init = 0.0
x_g = x_init + 0.25 + 0.32 + 0.25
y_g = y_init + 0

# Aux.
stop = 0                                    # Stop motor signal

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

mf_control_params = {
    'kv' : 0.8,                             # Vel. Gain         (sim = 0.8)       
    'kw' : 10,                               # Angular Vel. Gain (sim = 5)
    'xg' : 0.55,                             # Goal x coor.
    'yg' : -0.39,                             # Goal y coor.
    'l_width' : robot_prams['l_width']
}

circle_avoidance_params = {
    'R' : 0.15,
    'd_center' : 0.20
}

obs_algorithm_params = {
    'obs_method' : None,
    'margin' : 0.25
}
# ------------------------------------------------------------------


# Model
robot = model.model_diff_v1()
robot.initialize(robot_prams)

# Test Trajectory
# trajectory = [[0, 0], [0.2393462038024882, 0.3562770247256629], [0.28118389683772843, 0.3972041365035676], [0.3354990127324351, 0.41900526542701265], [0.394022567459391, 0.41836138724512935], [0.4478448803287615, 0.3953705265743392], [0.80, 0.15]]
# trajectory = [[0, 0], [0.2, 0], [0.4, 0.4]]
trajectory = [[0, 0], [x_g, y_g]]


# Obstacle avoidance Tr.
obs = obs_avoidance.circle_avoidance()
obs.initialize(circle_avoidance_params)
obs.angles = [180-a for a in range(15, 180, 15)]

# Obs. ALgorithm
obs_algorithm = obs_avoidance.obs_algorithm()
obs_algorithm_params['obs_method'] = obs
obs_algorithm.initialize(obs_algorithm_params)


# Controller
# policy = controllers.move_forward(mf_control_params)
pf_control_params['Tr'] = trajectory
policy = controllers.path_follower(pf_control_params)


# Communication
server = BluetoothMailboxServer()
# mbox = TextMailbox("greeting", server)

# The server must be started before the client!
print("waiting for connection...")
server.wait_for_connection()
print("connected!")

# Prepare the messages box
mbox = TextMailbox("greeting", server)


# In this program, the server waits for the client to send the first message
# and then sends a reply.
time.sleep(1)
mbox.send("PC ready to receive")
print("Now time to read")
print()

mbox.wait()
print(mbox.read())


# Store Vals.
state_list = []
vr_list = []
vl_list = []

counter = 100
first = 1
# while True :    
for i in range(0, 1050):
    # Start time
    start_time = time.time()

    # Ask for data
    mbox.send("PC")

    # Receive State
    mbox.wait()
    vel_right, vel_left, sensor_dist = interpreter.state_interpreter2(mbox.read(), vis=True)          # Split data
    vel_right, vel_left = op_funct.deg2m_two(vel_right, vel_left, robot.r_wheel)        # from deg/s to m/s
    if first :
        Ts = 0
        first = 0
    else:
        Ts = time.time() - cycle_time                                                       # compute sample time
        
    robot.Ts = Ts
    cycle_time = time.time()
    robot.step(vel_right, vel_left)                                                     # model step
    state = [robot.x, robot.y, robot.theta]
    
    # Obstacle Avoidance
    obs_algorithm.check_sensor(sensor_dist, policy.idx, state[0:2], trajectory)
    if obs_algorithm.controller_update :
        policy.Tr = obs_algorithm.Tr_obs
        policy.idx = 0

    # COMPUTE: Policy       
    vr, vl = policy.step( state, vis=False)
    vr, vl = op_funct.m2deg_two(vr, vl, robot.r_wheel)

    vr = max_vel*vr
    vl = max_vel*vl

    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance
    print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy.Tr[-1][0], policy.Tr[-1][1]) ) )
    if (distance <= tol_goal) :
        stop = 1 
        print()
        print("STOP", robot.x, robot.y)
        print()
            
    mbox.send(str(vr)+","+str(vl)+","+str(stop))
    if stop :
        break

    # Vis.
    print("Position = ", robot.x, robot.y, robot.theta)
    print("Ts = ", Ts)

    # Store
    state_list.append(state)
    vr_list.append(vr)
    vl_list.append(vl)

    # Wait for ACK
    mbox.wait()             
    print(mbox.read())

    # --- next_ev3 --- #

    # counter += 5
    time.sleep(T_wait)
    end_time = time.time()

    # Vis.
    print("*** Total time = ", end_time-start_time)






# fake signal
mbox.send("PC")
# Receive State
mbox.wait()
mbox.send(str(vr)+","+str(vl)+","+str(1))


fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_list]
state_y = [x[1] for x in state_list]

trajectory_x = [val[0] for val in policy.Tr]
trajectory_y = [val[1] for val in policy.Tr]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 

ax.grid(True)
ax.axis('equal')
plt.show() 



plt.plot(vr_list)
plt.plot(vl_list)
plt.show()