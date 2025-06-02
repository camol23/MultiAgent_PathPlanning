#!/usr/bin/env python3

import time 
from messaging import *

from utils_fnc import op_funct, interpreter
from control import model, controllers

# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------
T_wait = 0                                  # time.sleep per cycle
cycle_time = 0                              # time to reach the model_step
Ts = 0                                      # Sample time

tol_goal = 0.001                            # Stop signal when reach the goal [m]

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

mf_control_params = {
    'kv' : 0.8,                             # Vel. Gain         (sim = 0.8)       
    'kw' : 10,                               # Angular Vel. Gain (sim = 5)
    'xg' : 0.55,                             # Goal x coor.
    'yg' : -0.39,                             # Goal y coor.
    'l_width' : robot_prams['l_width']
}
# ------------------------------------------------------------------


# Model
robot = model.model_diff_v1()
robot.initialize(robot_prams)

# Controller
policy = controllers.move_forward(mf_control_params)


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

counter = 100
first = 1
while True :    
    # Start time
    start_time = time.time()

    # Ask for data
    mbox.send("PC")

    # Receive State
    mbox.wait()
    vel_right, vel_left = interpreter.state_interpreter(mbox.read(), vis=True)          # Split data
    vel_right, vel_left = op_funct.deg2m_two(vel_right, vel_left, robot.r_wheel)        # from deg/s to m/s
    if first :
        Ts = 0
        first = 0
    else:
        Ts = time.time() - cycle_time                                                       # compute sample time
        
    robot.Ts = Ts
    cycle_time = time.time()
    robot.step(vel_right, vel_left)                                                     # model step
    

    # COMPUTE: Policy    
    vr, vl = policy.step( [robot.x, robot.y, robot.theta], vis=True)
    vr, vl = op_funct.m2deg_two(vr, vl, robot.r_wheel)

    distance = op_funct.distance(robot.x, robot.y, policy.Tr[-1][0], policy.Tr[-1][1])    # compute distance
    if (distance <= tol_goal) :
        stop = 1 
            
    mbox.send(str(vr)+","+str(vl)+","+str(stop))
    if stop :
        break

    # Vis.
    print("Position = ", robot.x, robot.y, robot.theta)
    print("Ts = ", Ts)

    # Wait for ACK
    mbox.wait()             
    print(mbox.read())

    # --- next_ev3 --- #

    # counter += 5
    time.sleep(T_wait)
    end_time = time.time()

    # Vis.
    print("*** Total time = ", end_time-start_time)