#!/usr/bin/env python3

#
# The Circle coordinates are defined based on vector Ops.
#       not in circle and line intersention as in v1
#

import math
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import numpy as np

class obs_algorithm:
    def __init__(self):
        self.obs_method = None      # Trajectory generator
        self.id = 0

        # Sensor
        self.sensor_cm = math.inf
        self.active_flag = False

        # Aux.
        self.margin = 0.25
        self.idx_wp = 0
        self.controller_update = False # temporal

        self.Tr_obs = []
        self.Tr_og = []

        # Vals. For algorithm evolution
        self.wp_phase_2 = [-100, -100]        # Not Previous Collision
        self.wp_phase_3 = [-100, -100]        # Not Previous Collision
        self.point_phase_1 = []
        self.dist_phase_1 = -1              # Distance from robot to the first cricle point        
        self.mid_idx_cricle = 2             # Mid index in the angles array

        self.phase_indicator = 0
        # self.phase_0_flag = 0
        # self.phase_1_flag = 0               # First Circle generation           (Update Circle size and d_center)
        # self.phase_2_flag = 0               # 1: When reach the Circle top      (Recompute with the next WP)

        self.phase_1_conter = 1             # When reach 2 Start incrasing the Circle Radius
        self.state_phase_0 = []
        self.goal_coor = []

        self.counter_delay = 0              # Update trajectory just after to adjust a bit the heading
        self.delay_limit = 150 #80          # 80 Tested for the large
        
        self.trajectory_length = 5
        self.idx_output = 0
        self.large_obj_signal = 0

        self.aux_to_stop = 0


    def initialize(self, obs_algorithm_params, id=0):

        self.id = id
        self.obs_method = obs_algorithm_params['obs_method']
        self.margin = obs_algorithm_params['margin']
        
        self.active_flag = False

    def reset(self):
        self.obs_method.reset()
        self.idx_wp = 0
        self.controller_update = False      

        self.Tr_obs = []
        self.Tr_og = []

        # Vals. For algorithm evolution
        self.wp_phase_2 = [-100, -100]        # Not Previous Collision
        self.wp_phase_3 = [-100, -100]        # Not Previous Collision
        self.dist_phase_1 = -1                # Distance from robot to the first cricle point        
        self.mid_idx_cricle = 2               # Mid index in the angles array

        self.phase_indicator = 0        
        self.phase_1_conter = 1             # When reach 2 Start incrasing the Circle Radius
        self.state_phase_0 = []
        self.goal_coor = []

        self.counter_delay = 0              # Update trajectory just after to adjust a bit the heading
        self.delay_limit = 150 #80 
        
        self.trajectory_length = 5
        self.idx_output = 0
        self.large_obj_signal = 0

        self.aux_to_stop = 0


    def check_sensor(self, sensor_dist, current_idx, state, Tr, vis=False):
        '''
            Detect Obstacle and Generate Alternative path
        '''

        if len(self.Tr_og) > 0 :
            dist_to_goal = compute_distance(state[:2], self.Tr_og[self.idx_wp+1])
            if dist_to_goal < 2 :
                self.reset()
                print("Reset Obs. Algorithm")


        dist_to_phase_1 = compute_distance(state[:2], self.wp_phase_2)    
        dist_to_phase_3 = compute_distance(state[:2], self.wp_phase_3)
        phase_3_for_distance = 0    # flag to update in case_1 phase 3

        if dist_to_phase_1 <= 2 :
            self.phase_indicator = 2
            
        elif dist_to_phase_3 <= 2 :
            self.phase_indicator = 3
            # print("Pahse 3 SET ", self.id)


        if self.phase_indicator == 3 :                      
            # The top has been reached
            # Restart algorithm with a new WP
            # If it's necessary
            # Update working point (idx_wp)

            if (self.counter_delay == self.delay_limit) :
                
                increment_a = 1
                increment_b = 2
                
                # last point in Tr_og (No Update)
                if self.idx_wp == (len(self.Tr_og)-2) :
                    increment_a = 0
                    increment_b = 1                                   

                # if self.idx_wp != (len(self.Tr_og)-1) :
                dist_to_goal_wp = compute_distance(state[:2], self.Tr_og[self.idx_wp+increment_a])      # Current goal_WP 
                dist_to_next_wp = compute_distance(state[:2], self.Tr_og[self.idx_wp+increment_b])      # Next goal_WP 
                phase_3_for_distance = (0.8*dist_to_next_wp < dist_to_goal_wp)
                collision_phase_3 = (sensor_dist <= self.margin)
                if (phase_3_for_distance or collision_phase_3) and (len(self.Tr_og)>2) :
                    # Delete previous WP goal     
                    coor_to_remove = self.Tr_og[self.idx_wp+1]  # +0  
                    self.Tr_og.pop(self.idx_wp+1) # +0                                          
                    self.Tr_obs.remove(coor_to_remove)

                    # print("Inside for ", phase_3_for_distance, collision_phase_3)
                    # print("Delete  ", self.id, coor_to_remove, self.trajectory_length, len(self.Tr_obs))
                    # print("state ", current_idx, state)
                    # print("PHASE 3  ", self.id, self.Tr_obs[current_idx], self.Tr_og[self.idx_wp], self.Tr_og[self.idx_wp+1])
                    # print("section ", self.id, self.Tr_obs[current_idx:current_idx+4])
                    # print()
                    self.controller_update = True
                    self.idx_output = current_idx
                    self.counter_delay = 0

                    # ReDo algorithm
                    self.phase_indicator = 2
                    # self.aux_to_stop = True

                else:
                    self.controller_update = False
                # else:
                #     self.controller_update = False
            else:
                self.controller_update = False                                       


        if sensor_dist <= self.margin :                   

            if self.phase_indicator == 0 :                      # One execution (cheking)

                # self.active_flag = True                          # Obs. Detected
                self.idx_wp = current_idx
                self.Tr_og = copy.deepcopy(Tr)
                self.state_phase_0 = state

                self.validate_phase0_idx()

                self.goal_coor = Tr[self.idx_wp+1]                
                self.Tr_obs = self.generate_tr(self.state_phase_0, self.goal_coor)
                
                # idx update for current_pos 
                # Consequence to include robot_pos
                # self.idx_wp = self.idx_wp +1
                    
                self.controller_update = True
                self.phase_indicator = 1
                self.phase_1_conter = 1 #1
                # self.idx_output = self.idx_wp 
                self.idx_output = self.idx_wp + 1 # insert robot_pos in Tr_Obs
            
            elif self.phase_indicator == 1 :                                              
                # Still not reach the Circle
                # Move semi-perpendicular to Obst.

                if (self.counter_delay == self.delay_limit) :
                    
                    if self.phase_1_conter <= 2 : # 3
                        self.obs_method.d_center = (0.33)*self.phase_1_conter*self.obs_method.R                    


                    if self.phase_1_conter >= 3 : # 4
                        self.obs_method.d_center = (0.33)*self.phase_1_conter*self.obs_method.R
                        self.obs_method.R = (1.12)*self.obs_method.R                  
                        
                        # Probably is etter re-do the path for large object encounter
                        self.large_obj_signal = 1 
                        if self.phase_1_conter == 4 :
                            self.phase_indicator = 2

                    # Keep the previous WP_IDX  
                    #self.goal_coor = self.Tr_og[self.idx_wp+1] # +1 +0                             # Came from phase_0 and pahse_2
                    # if self.idx_wp <= 0 :
                    #     self.idx_wp = 1
                    # self.idx_wp = self.idx_wp - 1
                    self.Tr_obs = self.generate_tr(self.state_phase_0, self.goal_coor)
                        
                    self.controller_update = True
                    # self.idx_output = self.idx_wp 
                    self.idx_output = self.idx_wp + 1 # inser robot_pos in Tr_Obs

                    self.phase_1_conter = self.phase_1_conter + 1
                    self.counter_delay = 0               


                    if self.aux_to_stop :
                        self.delay_limit = -1000    
                    

            elif self.phase_indicator == 2 :                      
                # Circle before to reach the top
                # Update the reference line with current_pos 
                # and straigth along the heading angle
                # It's update state_0 & goal_coor

                if (self.counter_delay == self.delay_limit) :
                    
                    
                    ##self.goal_coor = Tr[self.idx_wp+1]
                    # if self.idx_wp <= 0 :
                    #     self.idx_wp = 1

                    # self.idx_wp = self.idx_wp - 1

                    # Update State                        
                    self.state_phase_0 = state
                    point = self.compute_point_oriented(state)
                    self.goal_coor = point

                    self.obs_method.reset()
                    self.Tr_obs = self.generate_tr(self.state_phase_0, point)                                    
                                                        
                    self.controller_update = True                    
                    self.counter_delay = 0       
                    # self.idx_output = self.idx_wp
                    self.idx_output = self.idx_wp + 1 # insert robot_pos in Tr_Obs

                    self.phase_indicator = 1
                    self.phase_1_conter = 1  # could be 1    


                    if self.aux_to_stop :
                        self.delay_limit = -1000     
                                         
            
            self.counter_delay = self.counter_delay + 1
        
        else:

            if phase_3_for_distance != 1 :
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
        
        self.validate_wp_circle_inside()

        # Take WP in the Circle Top 
        self.trajectory_length = len(trajectory)
        self.mid_idx_cricle =  math.floor( len(trajectory)/2)       # With the insert is not Odd
        self.wp_phase_3 = trajectory[self.mid_idx_cricle]  
        self.wp_phase_2 = trajectory[1]                             # The first point in the circle

        # trajectory.append(goal_coor)
        tr_init = self.Tr_og[:self.idx_wp+1]                        # [0 to idx)
        tr_end = self.Tr_og[self.idx_wp+1:]                         # [idx to last]
        output_trajectory = tr_init + trajectory + tr_end        
        
        # Correction for insert
        # if self.idx_wp >= (len(self.Tr_og)-2) :
        #     self.idx_wp = (len(self.Tr_og)-2)
        # else:
        #     self.idx_wp = self.idx_wp +1

        # print("State ", state)
        # print("Goal ", goal_coor)
        # print("IDX ", self.idx_wp)
        # print("Track = ", output_trajectory)
        # print()

        return output_trajectory


    def validate_wp_circle_inside(self):
        '''
            Make the correction to the WP 
            if it lands inside of the Circle
        '''

        eval_wp = self.Tr_og[self.idx_wp+1]

        eval_vector_x = eval_wp[0] - self.obs_method.x_center
        eval_vector_y = eval_wp[1] - self.obs_method.y_center
        eval_length = math.sqrt( eval_vector_x**2 + eval_vector_y**2 )
        
        if eval_length < self.obs_method.R :
            x = self.obs_method.x_center + self.obs_method.R*math.cos(self.obs_method.goal_angle)    # x
            y = self.obs_method.y_center + self.obs_method.R*math.sin(self.obs_method.goal_angle)    # y

            self.Tr_og[self.idx_wp+1] = [x, y]


    def compute_point_oriented(self, state):
        
        x = state[0] + 2*self.obs_method.mid_matgin*math.cos(state[2])
        y = state[1] + 2*self.obs_method.mid_matgin*math.sin(state[2])

        return [x, y]
    
    def validate_phase0_idx(self):
        '''
            If the following WP is between the obst.
            and the Circuference chooses the next one
        '''

        dist_current_wp = compute_distance(self.state_phase_0[:2], self.Tr_og[self.idx_wp+1])

        if dist_current_wp <= self.obs_method.mid_matgin :
            self.idx_wp = self.idx_wp + 1


    
    def wp_insede_correction(self):
        '''
            Move the Goal WP to the Intersection with the circle
        '''
        pass
        

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
        self.goal_angle = 0                 # Taken from current robot position

        # WPs
        self.angles = []

        # Result
        self.circle_wp = []

        self.circle_avoidance_params = None


    def initialize(self, circle_avoidance_params):
        # radius 
        # (Larger preferable)
        self.R = circle_avoidance_params['R']

        # Distance to the Circle Center
        # (small is preferable)
        self.d_center = circle_avoidance_params['d_center']        
        self.mid_matgin = circle_avoidance_params['mid_matgin']

        self.angles = [135, 112.5, 90, 67.5, 45]        # Itshould be odd size
        self.circle_avoidance_params = circle_avoidance_params

    def reset(self):
        
        self.R = self.circle_avoidance_params['R']
        self.d_center = self.circle_avoidance_params['d_center']        
        self.mid_matgin = self.circle_avoidance_params['mid_matgin']


    def compute_orthogonal(self):
        '''
         (1) Compute Orthogonal Vector
              (x, y) -> (-y, x)
        '''

        # Guideline Vector
        self.x_gl = self.xg - self.robot_x
        self.y_gl = self.yg - self.robot_y
        self.goal_angle = math.atan2(self.y_gl, self.x_gl) 

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
        
        self.x_mid = self.robot_x + (self.mid_matgin)*math.cos(self.goal_angle)
        self.y_mid = self.robot_y + (self.mid_matgin)*math.sin(self.goal_angle)



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
        # reset to recompute
        self.circle_wp = []

        self.robot_x = robot_pos[0]
        self.robot_y = robot_pos[1]

        self.xg = goal[0]
        self.yg = goal[1]
        
        self.compute_orthogonal()
        self.compute_center()

        # theta = math.atan2(self.yg, self.xg)       
        theta = self.goal_angle

        # circle_wp = []
        # self.angles = [135, 112.5, 90, 67.5, 45]
        # self.angles = [180, 170, 160, 150, 140, 135, 112.5, 90, 67.5, 45, 30, 20, 10]
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



def detect_obs_sim(state, obs_list, detect_obs_params, momentary_signal=0):
    '''
        Funtunction to Dectect the obstacle in a 
        certain range during simulation        

        obstacles: (x_botton_left, y_botton_left, width, height)

        Output: Boolean
    '''

    collision_flag = 0
    angle_range = detect_obs_params['angle_range']
    ray_length = detect_obs_params['ray_length']
    num_rays = 11

    # Rays orientation
    start_angle = state[2] + angle_range
    finish_angle = state[2] - angle_range
    angles = np.linspace(start_angle, finish_angle, num_rays, False)    

    # Sensor Projection
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    rays_x = state[0] + ray_length*cos_angles
    rays_y = state[1] + ray_length*sin_angles


    for i in range(0, len(obs_list)):
        #  y_botton < X > y_up  (X.shape( num_points ))
        # mask_collision = (obs_list[i][1] < rays_y) & (rays_y < (obs_list[i][3]) )                     # last is a coordinate 
        mask_collision = (obs_list[i][1] < rays_y) & (rays_y < (obs_list[i][1] + obs_list[i][3]) )      # Last is heigth
        
        # x_bottom < x_fix < x_up
        # mask_columns = (obs_list[i][0] < rays_x) & (rays_x < (obs_list[i][2]) )
        mask_columns = (obs_list[i][0] < rays_x) & (rays_x < (obs_list[i][0] + obs_list[i][2]) )
        mask_collision = mask_collision & mask_columns

        # Just One detection
        num_collisitons = np.sum(mask_collision)

        if num_collisitons > 0 :
            collision_flag = 1            
            break
        else:
            collision_flag = 0

    
    if momentary_signal == 3 :
        print("Sensor ----")
        print("angles ", np.degrees(angles))
        print("State ", state)
        print("num_collisitons ", num_collisitons)
        print()

    return collision_flag

    



def compute_distance(point_a, point_b):
    dist = math.sqrt( (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)

    return dist



