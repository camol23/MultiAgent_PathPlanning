#!/usr/bin/env python3

#
# The Circle coordinates are defined based on vector Ops.
#       not in circle and line intersention as in v1
#

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class obs_algorithm:
    def __init__(self):
        self.obs_method = None      # Trajectory generator

        # Sensor
        self.sensor_cm = math.inf
        self.active_flag = False

        # Aux.
        self.margin = 0.25
        self.idx_wp = 0
        self.controller_update = False # temporal

        self.Tr_obs = []

    def initialize(self, obs_algorithm_params):

        self.obs_method = obs_algorithm_params['obs_method']
        self.margin = obs_algorithm_params['margin']

        self.active_flag = False


    def check_sensor(self, sensor_dist, current_idx, state, Tr, vis=False):
        '''
            Detect Obstacle and Generate Alternative path
        '''
        if sensor_dist <= self.margin :
            if self.active_flag == False :

                self.active_flag = True                         # Obs. Detected
                self.idx_wp = current_idx

                goal_coor = Tr[self.idx_wp+1]
                self.Tr_obs = self.generate_tr(state, goal_coor)

                self.controller_update = True
        else:
            self.controller_update = False
            

        # Vis
        if vis :
            print("Dinstance Sensor = ", sensor_dist, self.active_flag)


    def generate_tr(self, state, goal_coor):
        '''
            Generate a new Trajectory
        ''' 
        current_state = [state[0], state[1]]

        self.obs_method.compute_tr(goal_coor, current_state)
        trajectory = self.obs_method.circle_wp
        trajectory.insert(0, current_state)
        trajectory.append(goal_coor)

        return trajectory


class circle_avoidance:
    def __init__(self):
        
        self.robot_x = 0
        self.robot_y = 0

        # Goal Coor.
        self.xg = 0
        self.yg = 0

        # Axis for Circle center
        self.x_mid = 0
        self.y_mid = 0 

        # Orthogonal Vector
        self.x_ort = 0
        self.y_ort = 0

        # radius 
        self.R = 15 

        # Distance to the Circle Center
        self.d_center = 20
        self.mid_matgin = 40

        # Guideline Vector
        self.x_gl = 0
        self.y_gl = 0

        # WPs
        self.angles = []

        # Result
        self.circle_wp = []



    def initialize(self, circle_avoidance_params):
        # radius 
        # (Larger preferable)
        self.R = circle_avoidance_params['R']

        # Distance to the Circle Center
        # (small is preferable)
        self.d_center = circle_avoidance_params['d_center']        

        self.angles = [135, 112.5, 90, 67.5, 45]


    def compute_orthogonal(self):
        '''
         (1) Compute Orthogonal Vector
              (x, y) -> (-y, x)
        '''

        # Guideline Vector
        self.x_gl = self.xg - self.robot_x
        self.y_gl = self.yg - self.robot_y

        # 90 degrees Rotation
        x_ort = -self.y_gl
        y_ort = self.x_gl

        # Unit vector
        norm_ort = math.sqrt((x_ort**2) + (y_ort**2))
        self.x_ort = x_ort/norm_ort
        self.y_ort = y_ort/norm_ort


        # mid. point 
        # (It can be adjusted to move closer to the Obs.)
        # self.x_mid = self.robot_x + (self.x_gl/2)
        # self.y_mid = self.robot_y + (self.y_gl/2)

        self.x_mid = self.robot_x + (self.mid_matgin)
        self.y_mid = self.robot_y + (self.mid_matgin)


    def compute_center(self):
        '''

             (2) Compute Circle Center
                d_center = norm( (h, k), (x_mid, y_mid) )
                Small (d_center) should give a proper path
            
            Later can be compare the distance to the (xg,yg)
            with the distance to the next wp (choose the shortest)        

        '''

        self.x_ort = self.d_center*self.x_ort
        self.y_ort = self.d_center*self.y_ort

        # Orthogonal vector translation
        x1_ort = self.x_ort + self.x_mid
        y1_ort = self.y_ort + self.y_mid

        # Choose center (90 or -90 [deg])
        self.x_center = x1_ort
        self.y_center = y1_ort

        # distance_center_1 = math.sqrt( (self.x_mid-self.x_center)**2 +  (self.y_mid-self.y_center)**2 )


    def compute_tr(self, goal, robot_pos):
        '''
            (3) Circle WP points
        '''

        self.robot_x = robot_pos[0]
        self.robot_y = robot_pos[1]

        self.xg = goal[0]
        self.yg = goal[1]
        
        self.compute_orthogonal()
        self.compute_center()

        theta = math.atan2(self.yg, self.xg)       

        # circle_wp = []
        # angles = [135, 112.5, 90, 67.5, 45]
        for angle in self.angles:
            wp_x = self.x_center + self.R*math.cos(theta + math.radians(angle))    # x
            wp_y = self.y_center + self.R*math.sin(theta + math.radians(angle))    # y

            self.circle_wp.append([wp_x, wp_y])

            


# Visualization
    def vis(self):
        '''
            Vis. Construction elements            
        '''    
        
        # trajectory line
        mt = self.yg/self.xg
        bt = self.yg - mt*self.xg
        x_end = 2*self.xg
        y_end = mt*x_end + bt


        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1) 
        
        trajectory_x = [val[0] for val in self.circle_wp]
        trajectory_x.insert(0, self.robot_x)
        trajectory_x.append(self.xg)
        trajectory_x.insert(0, -self.xg)
        trajectory_x.append(x_end)

        trajectory_y = [val[1] for val in self.circle_wp]
        trajectory_y.insert(0, self.robot_y)
        trajectory_y.append(self.yg)
        trajectory_y.insert(0, -self.yg)
        trajectory_y.append(y_end)

        
        ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
        ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

        ax.plot(trajectory_x[1], trajectory_y[1], color ='yellow', linestyle='--')

        # Circle
        ax.scatter(self.x_center, self.y_center, c='blue', alpha=0.5, linewidths=0.5)
        ax.add_patch(Circle( (self.x_center, self.y_center), radius=self.R, ec='blue', fill=False, linestyle='--') )
    
        # ax.set_title('All Paths - Shortest '+ str(int(sorted_cost[0])) + " - second = "+ str(int(sorted_cost[1])) ) 
        
        ax.grid(True)
        ax.axis('equal')
        plt.show() 


    def vis_multi(self, ax):
        '''
            Vis. Construction elements            
        '''    
        
        # trajectory line
        mt = self.yg/self.xg
        bt = self.yg - mt*self.xg
        x_end = 2*self.xg
        y_end = mt*x_end + bt


        # fig = plt.figure() 
        # ax = fig.add_subplot(1, 1, 1) 
        
        trajectory_x = [val[0] for val in self.circle_wp]
        trajectory_x.insert(0, self.robot_x)
        trajectory_x.append(self.xg)
        trajectory_x.insert(0, -self.xg)
        trajectory_x.append(x_end)

        trajectory_y = [val[1] for val in self.circle_wp]
        trajectory_y.insert(0, self.robot_y)
        trajectory_y.append(self.yg)
        trajectory_y.insert(0, -self.yg)
        trajectory_y.append(y_end)

        
        ax.plot(trajectory_x, trajectory_y, color ='tab:red') 
        ax.scatter(trajectory_x, trajectory_y, c='red', alpha=0.5, linewidths=0.5)

        ax.plot(trajectory_x[1], trajectory_y[1], color ='yellow', linestyle='--')

        # Circle
        ax.scatter(self.x_center, self.y_center, c='blue', alpha=0.5, linewidths=0.5)
        ax.add_patch(Circle( (self.x_center, self.y_center), radius=self.R, ec='blue', fill=False, linestyle='--') )
    
        # ax.set_title('All Paths - Shortest '+ str(int(sorted_cost[0])) + " - second = "+ str(int(sorted_cost[1])) ) 
        
        ax.grid(True)
        ax.axis('equal')
        # plt.show() 


















