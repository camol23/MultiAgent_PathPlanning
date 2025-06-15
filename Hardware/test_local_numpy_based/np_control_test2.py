#!/usr/bin/env python3

import time 
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, MoveDifferential, SpeedDPS
from ev3dev2.sensor import INPUT_2
from ev3dev2.sensor.lego import UltrasonicSensor
from ev3dev2.sensor.lego import GyroSensor

from ev3dev2.wheel import EV3EducationSetTire
from ev3dev2.wheel import EV3Tire

from utils_fnc import op_funct, interpreter
from control import model, controllers, obs_avoidance, kalman
from logging_data import store_data


print("Intro -> kw_input (5)")
kw_input = float(input())

print("Intro -> k_rot (3.5)")
k_rot = float(input())

print("kw_input ", kw_input)
print("k_rot ", k_rot)



# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------
T_wait = 0                                  # time.sleep per cycle
cycle_time = 0                              # time to reach the model_step
Ts = 0                                      # Sample time

# max_vel = 0.0977384
tol_goal = 0.1                            # Stop signal when reach the goal [m]

# Trajectory
x_init = 0.25
y_init = 0.0
# x_g = x_init + 0.25 + 0.32 + 0.25
# y_g = y_init + 0

x_g = 1
y_g = 0

# Aux.
stop = 0                                    # Stop motor signal

x_init_robot = 0
y_init_robot = 0

robot_prams = {
    'x_init' : x_init_robot,
    'y_init' : y_init_robot,
    'theta_init' : 0,

    'l_width' : 0.105,                      # meters
    'r_width' : 0.056/2,                    # meters
    'Ts' : Ts
}

pf_control_params = {
    'kv' : 0.1,               # Vel. Gain         (sim = 0.8) works = 0.1
    # 'kw' : 5,                 # Angular Vel. Gain (sim = 5) works = 5
    # 'k_rot' : 3.5,              # Heading Sensitive (sim = 5) works = 5
    'kw' : kw_input,
    'k_rot' : k_rot,

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
    'd_center' : 0.20,
    'mid_matgin' : 25 + 7        # 7: is the 4th part of the circle diameter       
}

obs_algorithm_params = {
    'obs_method' : None,
    'margin' : 0.25
}


ekf_params = {
    'H_rows' : 4,
    'H_colmns' : 3,
    'X0' : [x_init_robot, y_init_robot, 0],
    'P0_factor' : 10,
    'Q_sigma_list' : [36e-6, 36e-6, 2*36e-6],           # Var. Prossesing Noise
    'R_sigma_list' : [36e-6, 36e-6, 2*36e-6, (140e-6)/1000]    # Var. Measurement Noise
}
# ------------------------------------------------------------------

# --------------------------- EV3 ----------------------------------

# Motors
# motor_right = LargeMotor(OUTPUT_C)
# motor_left = LargeMotor(OUTPUT_B)
# motor_right.reset()
# motor_left.reset()

# gyro = GyroSensor() 
# gyro.calibrate()                    
# gyro.reset()


ultrasonic_sensor = UltrasonicSensor(INPUT_2)

size_w = 87 
mdiff = MoveDifferential(OUTPUT_B, OUTPUT_C, EV3Tire, size_w) # 87 with speed = 300
mdiff.gyro = GyroSensor()
mdiff.gyro.calibrate()      
mdiff.odometry_start(theta_degrees_start=0.0, x_pos_start=0.0, y_pos_start=0.0) # Not accurate

# ------------------------------------------------------------------

# Model
robot = model.model_diff_v1()
robot.initialize(robot_prams)

# Test Trajectory
# trajectory = [[0, 0], [0.2393462038024882, 0.3562770247256629], [0.28118389683772843, 0.3972041365035676], [0.3354990127324351, 0.41900526542701265], [0.394022567459391, 0.41836138724512935], [0.4478448803287615, 0.3953705265743392], [0.80, 0.15]]
# trajectory = [[0, 0], [0.2, 0], [0.4, 0.4]]


# trajectory = [[x_init_robot, y_init_robot], [x_g, y_g]] # Tested
trajectory = [[x_init_robot, y_init_robot], [0.4, 0.5], [x_g, y_g]]


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


# EKF
ekf = kalman.kalman()
ekf.initialization(ekf_params)

# Communication
# Inlude a Beep



# Store Vals.
state_list = []
state_ekf_list = []
gyro_list = []
vr_list = []
vl_list = []

counter = 100
first = 1

# for i in range(0, 1050):
for i in range(0, 120):
    # Start time
    start_time = time.time()


    # Receive State
    # vel_right = motor_right.speed
    # vel_left = motor_left.speed
    vel_right = mdiff.right_motor.speed
    vel_left = mdiff.left_motor.speed

    print("Vel. Read  (r, l) = ", vel_right, vel_left)
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

    angle_from_gyro = mdiff.gyro.angle
    # gyro_list.append(angle_from_gyro)
    gyro_angle = model.wrapped_angle_360_deg(-angle_from_gyro)      
    gyro_angle_rad = op_funct.degToRad(gyro_angle)
    gyro_list.append(gyro_angle)

    y = state + [gyro_angle_rad]
    u = [robot.v, robot.w]
    ekf.compute(y, u, Ts)
    state_ekf_list.append(ekf.X_hat)
    
    # Obstacle Avoidance
    sensor_dist_1 = ultrasonic_sensor.distance_centimeters    
    sensor_dist_1 = sensor_dist_1/100
    # obs_algorithm.check_sensor(sensor_dist_1, policy.idx, state[0:2], trajectory)
    # if obs_algorithm.controller_update :
    #     policy.Tr = obs_algorithm.Tr_obs
    #     policy.idx = 0

    # COMPUTE: Policy       
    
    # vr, vl = policy.step( state, vis=False)
    vr, vl = policy.step( ekf.X_hat, vis=False)
    vr, vl = op_funct.m2deg_two(vr, vl, robot.r_wheel)

    print("Vel. Policy  (r, l) = ", vr, vl)
    # vr = max_vel*vr
    # vl = max_vel*vl

    # vr, vl = op_funct.m2deg_two(vr, vl, robot.r_wheel)
    # print("Vel. Policy  (r, l) = ", vr, vl)

    # distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance
    distance = op_funct.distance(ekf.X_hat[0], ekf.X_hat[1], policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance
    print("Distance to the Goal = {} - Goal = {} ".format( distance, (policy.Tr[-1][0], policy.Tr[-1][1]) ) )
    if (distance <= tol_goal) :
        stop = 1 
        print()
        print("STOP", robot.x, robot.y)
        print()
    

    vr = max( min(300, float(vr) ), 0)
    vl = max( min(300, float(vl) ), 0)

    # For test (No Controller)       
    # vr, vl = 300, 300

    # mbox.send(str(vr)+","+str(vl)+","+str(stop))
    # motor_right.speed_sp = vr
    # motor_left.speed_sp = vl
    # motor_right.run_forever()
    # motor_left.run_forever() 
    mdiff.on(SpeedDPS(vl), SpeedDPS(vr)) # (left, right)   
    if stop :
        break

    # Vis.
    print("Position = ", robot.x, robot.y, robot.theta)
    print("Position EKF = ", ekf.X_hat)
    print("Ts = ", Ts)
    print()

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
    print()


print()
print("End ", i)
# Brake
# motor_right.stop()
# motor_left.stop()
mdiff.stop(brake=True)
mdiff.stop()

print("Position = "+ str(robot.x) + " " + str(robot.y) + " "+ str(robot.theta))
print("Position EKF = ", ekf.X_hat)
print("Gyro Last (rad, deg_wrapped) = ", mdiff.gyro.angle, gyro_list[-1])
print("End position internal Odo. EV3 = ", mdiff.x_pos_mm, mdiff.y_pos_mm, mdiff.theta)
mdiff.odometry_stop()

# with open("./logging_data/dist_test0.txt", "a") as f:
#   f.write("\n")
#   f.write("New test\n")
#   f.write("Position = "+ str(robot.x) + " " + str(robot.y) + " "+ str(robot.theta) +"\n")



# Prepaer Data to be saved
dict_data = {
    'state_list' : state_list,
    'state_ekf_list' : state_ekf_list,
    'gyro_list' : gyro_list,
    'vr_list' : vr_list,
    'vl_list' : vl_list,
    'policy_Tr' : policy.Tr,
    'trajectory' : trajectory
}

# path = "./logging_data/time_data_exp/"
path = '/home/robot/test_local_numpy_based/logging_data/measures/'
name = "control_ekf_test"
store_data.save_pickle(name, dict_data, path)


