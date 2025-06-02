#!/usr/bin/env python3

import rpyc
import time 

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
tol_goal = 0.1                            # Stop signal when reach the goal [m]

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

# --------------------------- EV3 ----------------------------------

conn_ev1 = rpyc.classic.connect('10.16.12.26') # ev3dev
conn_ev2 = rpyc.classic.connect('10.16.12.25') # ticket

# import ev3dev2 on the remote ev3dev device
ev3dev2_1_motor_C = conn_ev1.modules['ev3dev2.motor']
ev3dev2_1_motor_B = conn_ev1.modules['ev3dev2.motor']

ev3dev2_2_motor = conn_ev2.modules['ev3dev2.motor']

# Use the LargeMotor and TouchSensor on the remote ev3dev device
motor_right_ev1 = ev3dev2_1_motor_C.LargeMotor(ev3dev2_1_motor_C.OUTPUT_C)
motor_left_ev1  = ev3dev2_1_motor_B.LargeMotor(ev3dev2_1_motor_B.OUTPUT_B)
motor_right_ev2 = ev3dev2_2_motor.LargeMotor(ev3dev2_2_motor.OUTPUT_C)

# Sensor
ev3dev2_1_sensor = conn_ev1.modules['ev3dev2.sensor']
ev3dev2_1_sensor_lego = conn_ev1.modules['ev3dev2.sensor.lego']

dist_sensor_ev1 = ev3dev2_1_sensor_lego.UltrasonicSensor(ev3dev2_1_sensor.INPUT_2)


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
# Inlude a Beep
# motor_right_ev1.run_forever(speed_sp=0)
# motor_right_ev2.run_forever(speed_sp=0)


# Store Vals.
state_list = []
vr_list = []
vl_list = []

counter = 100
first = 1

# for i in range(0, 1050):
for i in range(0, 200):
    # Start time
    start_time = time.time()


    # Receive State
    vel_right = motor_right_ev1.speed
    vel_left = motor_left_ev1.speed
    print(vel_right, vel_left)
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
    sensor_dist_1 = dist_sensor_ev1.distance_centimeters
    sensor_dist_1 = sensor_dist_1/100
    # obs_algorithm.check_sensor(sensor_dist_1, policy.idx, state[0:2], trajectory)
    # if obs_algorithm.controller_update :
    #     policy.Tr = obs_algorithm.Tr_obs
    #     policy.idx = 0

    # COMPUTE: Policy       
    vr, vl = policy.step( state, vis=False)
    vr, vl = op_funct.m2deg_two(vr, vl, robot.r_wheel)

    # vr = max_vel*vr
    # vl = max_vel*vl

    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance
    print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy.Tr[-1][0], policy.Tr[-1][1]) ) )
    if (distance <= tol_goal) :
        stop = 1 
        print()
        print("STOP", robot.x, robot.y)
        print()
            
    
    vr = max( min(200, float(vr) ), 0)
    vl = max( min(200, float(vl) ), 0)
    # mbox.send(str(vr)+","+str(vl)+","+str(stop))
    motor_right_ev1.run_forever(speed_sp=vr)
    motor_left_ev1.run_forever(speed_sp=vl)    
    if stop :
        break

    # Vis.
    print("Position = ", robot.x, robot.y, robot.theta)
    print("Ts = ", Ts)

    # Store
    state_list.append(state)
    vr_list.append(vr)
    vl_list.append(vl)

    # --- next_ev3 --- #

    # counter += 5
    time.sleep(T_wait)
    end_time = time.time()

    # Vis.
    print("*** Total time = ", end_time-start_time)


# Brake
motor_right_ev1.stop()
motor_left_ev1.stop()



# Results 
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