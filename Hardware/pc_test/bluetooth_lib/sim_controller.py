#!/usr/bin/env python3

from control import obs_avoidance, controllers, model

import matplotlib.pyplot as plt


# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------

# Goal
xg = 25 + 30 + 25
yg = 15

# Current Pos
robot_x = 0
robot_y = 0

# Sim
Ts = 0.1

circle_avoidance_params = {
    'R' : 15,
    'd_center' : 20
}

pf_control_params = {
    'kv' : 0.8,               # Vel. Gain         (sim = 0.8)
    'kw' : 5,                 # Angular Vel. Gain (sim = 5)
    'k_rot' : 5,              # Heading Sensitive (sim = 5)

    # trajectory
    'Tr' : [],
    
    # Aux. 
    'l_width' : 0.105,        # robot width (0.105)
    'Ts' : Ts
}

robot_prams = {
    'x_init' : 0,
    'y_init' : 0,
    'theta_init' : 0,

    'l_width' : 0.105,                      # meters
    'r_width' : 0.056/2,                    # meters
    'Ts' : Ts
}
# ------------------------------------------------------------------


obs = obs_avoidance.circle_avoidance()
obs.initialize(circle_avoidance_params)

# Test
obs.compute_tr([xg, yg], [robot_x, robot_y])
trajectory = obs.circle_wp
trajectory.insert(0, [robot_x, robot_y])
trajectory.append([xg, yg])
# print("Trajectory = ", trajectory)

# Visualization
#obs.vis()

# Controller
pf_control_params['Tr'] = trajectory
control = controllers.path_follower(pf_control_params)
print("Trajectory = ", control.Tr)

# Model
robot = model.model_diff_v1()
robot.initialize(robot_prams)
state = [robot.x, robot.y, robot.theta]


# Store vals.
state_list = []
vr_list = []
vl_list = []

for i in range(0, 1000):


    vel_right, vel_left = control.step(state)

    # Sim.
    robot.step(vel_right, vel_left)
    state = [robot.x, robot.y, robot.theta]

    # Storing
    state_list.append(state)
    vr_list.append(vel_right)
    vl_list.append(vel_left)

    # Vis.
    # print("Idx = ", control.idx, control.v)









fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_list]
state_y = [x[1] for x in state_list]

trajectory_x = [val[0] for val in control.Tr]
trajectory_y = [val[1] for val in control.Tr]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 

ax.grid(True)
ax.axis('equal')
plt.show() 



plt.plot(vr_list)
plt.plot(vl_list)
plt.show()

# state_x = [x[0] for x in state_list]
# state_y = [x[1] for x in state_list]
# plt.plot(state_x, state_y)
# plt.show()