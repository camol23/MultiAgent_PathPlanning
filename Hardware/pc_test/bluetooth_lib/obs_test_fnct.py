#!/usr/bin/env python3

from control import obs_avoidance
import matplotlib.pyplot as plt



# ------------------------------------------------------------------
#       Settings
# ------------------------------------------------------------------

# Goal
xg = 25 + 30 + 25 + 20
yg = 0

# Current Pos
robot_x = 0
robot_y = 0

circle_avoidance_params = {
    'R' : 15,
    'd_center' : 10
}
# ------------------------------------------------------------------


obs = obs_avoidance.circle_avoidance()
obs.initialize(circle_avoidance_params)


# Test
obs.compute_tr([xg, yg], [robot_x, robot_y])


# Visualization
# obs.vis() # works



# ----------------------- To Vis sequence -------------
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) 
obs.vis_multi(ax)

obs.circle_wp = []
# robot_x = 4
# robot_y = 10
obs.d_center = 50
obs.compute_tr([xg, yg], [robot_x, robot_y])
obs.vis_multi(ax)

obs.circle_wp = []
# robot_x = 10
# robot_y = 20
obs.d_center = 100
obs.compute_tr([xg, yg], [robot_x, robot_y])
obs.vis_multi(ax)

plt.show()