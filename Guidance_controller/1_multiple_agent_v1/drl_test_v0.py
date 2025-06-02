
import tensorflow as tf
import matplotlib.pyplot as plt

from control import obs_avoidance, controllers, model
from DRL2 import observations_module
from utils_fnc import op_funct

import actor_tg4_tf




# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------

# Goal
# xg = 25 + 30 + 25
xg = 20
yg = 0
goal_coor = [xg, yg]

# Agent
max_speed = 2 #4.87

# Current Pos
robot_x = 0
robot_y = 0

# Sim
Ts = 0.1

module_obs_params = {
    'scale_pos' : 10,
    'scale_obs' : 4
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

# Model
robot = model.model_diff_v1()
robot.initialize(robot_prams)
state = [robot.x, robot.y, robot.theta]
vels = [robot.Vx, robot.Vy, robot.w]

# Agent
policy = actor_tg4_tf.load_model()

# Obs. Module
obs_module = observations_module.module_observations()
obs_module.initialize(module_obs_params)
obs_module.WP0 = [robot_prams['x_init'], robot_prams['y_init']]

# Store vals.
state_list = []
vr_list = []
vl_list = []
vx_list = []
vy_list = []

state_x = []
state_y = []

fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 
trajectory_x = [0, goal_coor[0]]
trajectory_y = [0, goal_coor[1]]
ax.plot(trajectory_x, trajectory_y, color ='tab:red')
plt.axis([-10, 150, -10, 150]) 
ax.grid(True)
ax.axis('equal')
plt.show(block=False) 

for i in range(0, 500):

    observations = obs_module.compute(state, goal_coor, vels)
    obs_input = tf.convert_to_tensor(observations)
    action = policy.predict(obs_input, verbose=0)
    
    vel_right = action[0][0]
    vel_left = action[0][1]

    # Adjust to training Val. (Test)
    vel_right = max_speed*vel_right
    vel_left = max_speed*vel_left

    # Sim.
    robot.step(vel_right, vel_left)
    state = [robot.x, robot.y, robot.theta]
    vels = [robot.Vx, robot.Vy, robot.w]

    # Storing
    state_list.append(state)
    vr_list.append(vel_right)
    vl_list.append(vel_left)
    vx_list.append(vels[0])
    vy_list.append(vels[1])

    state_x.append(state[0])
    state_y.append(state[1])
    ax.plot(state_x, state_y, color ='blue') 
    plt.draw()
    plt.pause(0.001)
    
    # End
    dist_goal = op_funct.distance(state[0], state[1], goal_coor[0], goal_coor[1])
    if dist_goal <= 2.5 :
        print("Stop in i = ", i)
        break
    




fig = plt.figure() 
ax = fig.add_subplot(1, 1, 1) 


state_x = [x[0] for x in state_list]
state_y = [x[1] for x in state_list]

trajectory_x = [0, goal_coor[0]]
trajectory_y = [0, goal_coor[1]]


ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

ax.plot(state_x, state_y, color ='blue') 

ax.grid(True)
ax.axis('equal')
plt.show() 



plt.plot(vr_list)
plt.plot(vl_list)
plt.show()