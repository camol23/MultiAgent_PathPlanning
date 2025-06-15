import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from utils_fnc import op_funct

'''
    Test Coordinate transformation
    ('play ground' is not interactive - just visualization)
'''


# Input Coor.
xr = 0.10
yr = 0.10
theta_r = math.radians(15)

xg = 1.00 
yg = 1.00

arbritary = [0.30, 0.20]

detection_distance = 0.25

# Frame are the Transformed Coor.
# Coor. for the DRL
# Frame Data 
frame_scale = 0.02
frame_size = 20                         # complete size is frame_size*2
r_circ = (frame_size*frame_scale) - 0.02   # real area

obst_r_frame = 4
obst_r = (obst_r_frame*frame_scale)

# DRL map
x0 = None               # Center in real Coordinates
y0 = None   

corners = []

xg_frame = None
yg_frame = None
xr_frame = None
yr_frame = None


# Algorithm

# Compute obstacle Center
x0 = xr + (detection_distance + obst_r)*math.cos(theta_r)
y0 = yr + (detection_distance + obst_r)*math.sin(theta_r)

# Compute frame Corners (rela Coor.)
angle_corner = math.radians(45)

h_in_frame = math.sqrt( 2*(frame_size*frame_scale)**2 )
for i in range(0, 4):
    xi = x0 + (h_in_frame)*math.cos(angle_corner)
    yi = y0 + (h_in_frame)*math.sin(angle_corner)

    corners.append([xi, yi])
    angle_corner = angle_corner + math.radians(90)
    print(math.degrees(angle_corner), math.cos(math.radians(angle_corner)))
    print()

# Compute Subgoal
l_points = [[xr, yr], [xg, yg]]
circle_coor = [x0, y0]

point1, point2, delta = op_funct.circle_line_intersection(l_points, circle_coor, r_circ)
print("Circle-Line Inter.")
print("Points = ", point1, point2, delta)
print()

## choose subgoal
v1 = [ point1[0]-xr, point1[1]-yr]
v2 = [ point2[0]-xr, point2[1]-yr]
v_ref = [xg - xr, yg - yr]

ang_1 = math.atan2(v1[1], v1[0])
ang_2 = math.atan2(v2[1], v2[0])
ang_ref = math.atan2(v_ref[1], v_ref[0])

if ang_1 == ang_ref :
    sub_goal = point1
else:
    sub_goal = point2


# Coor. Transformation
xr_frame, yr_frame = op_funct.trans_coor([xr, yr], [x0, y0], frame_scale)
xg_sub_frame, yg_sub_frame = op_funct.trans_coor(sub_goal, [x0, y0], frame_scale)

# Guide Pointes (visualization)
xg_frame, yg_frame = op_funct.trans_coor([xg, yg], [x0, y0], frame_scale)
corners_frame = []
for corner_i in corners:
    xi_frame, yi_frame = op_funct.trans_coor(corner_i, [x0, y0], frame_scale)
    corners_frame.append([xi_frame, yi_frame])

arbritary_frame = op_funct.trans_coor(arbritary, [x0, y0], frame_scale)

# Visualization
fig = plt.figure() 
axes = fig.add_subplot(1, 2, 1) 


axes.plot([xr, xg], [yr, yg], color ='blue') 
axes.scatter(sub_goal[0], sub_goal[1], c='blue', alpha=0.5, linewidths=0.5)
axes.scatter(x0, y0, c='red', alpha=1, linewidths=1)

# frame
w_real_frame = abs(corners[0][0] - corners[1][0])
h_real_frame = abs(corners[0][1] - corners[2][1])
axes.add_patch(Rectangle((corners[2][0], corners[2][1]), w_real_frame, h_real_frame, fill=False, ec='black'))
print('w_real_frame', w_real_frame)
print('h_real_frame', h_real_frame)

axes.add_patch(Circle((x0, y0), radius=r_circ, alpha=0.3))
axes.add_patch(Circle((x0, y0), radius=obst_r, alpha=0.3, fill=False, ec='black', ls='--'))


axes.scatter(arbritary[0], arbritary[1], c='yellow', alpha=1, linewidths=1)

axes.grid(True)
axes.axis('equal')

axes2 = fig.add_subplot(1, 2, 2)
axes2.plot([xr_frame, xg_frame], [yr_frame, yg_frame], color ='blue') 
axes2.scatter(xg_sub_frame, yg_sub_frame, c='blue', alpha=0.5, linewidths=0.5)
axes2.scatter(0, 0, c='red', alpha=1, linewidths=1)

# frame
w_real_frame = abs(corners_frame[0][0] - corners_frame[1][0])
h_real_frame = abs(corners_frame[0][1] - corners_frame[2][1])
axes2.add_patch(Rectangle((corners_frame[2][0], corners_frame[2][1]), w_real_frame, h_real_frame, fill=False, ec='black'))

axes2.add_patch(Circle((0, 0), radius=frame_size, alpha=0.3))
axes2.add_patch(Circle((0, 0), radius=obst_r_frame, alpha=0.3, fill=False, ec='black', ls='--'))

axes2.scatter(arbritary_frame[0], arbritary_frame[1], c='yellow', alpha=1, linewidths=1)


axes2.grid(True)
axes2.axis('equal')
plt.show() 

