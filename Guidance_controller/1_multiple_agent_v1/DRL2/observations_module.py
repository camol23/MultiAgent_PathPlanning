#
# Translate the Model robot data in Agent Observations
#

import math
import numpy as np


class module_observations:
    def __init__(self):
        
        # State
        self.Xe = 0
        self.Ye = 0
        self.Th_e = 0
        self.Th   = 0

        self.Vx = 0
        self.Vy = 0
        self.W = 0

        self.guideline_dist = 0
        self.obs_dist = 0

        # Variables
        self.WP0 = (0, 0)           # Should be set based in the current waypoint

        # Parameters
        self.scale_pos = 0          # Scaled the robot position (trained with 10)
        self.scale_obs = 0


    def initialize(self, module_obs_params):
        
        self.scale_pos = module_obs_params['scale_pos']
        self.scale_obs = module_obs_params['scale_obs']


    def compute(self, state, goal, vels, obst_flag=False, obst_coor=[0,0], no_reshape_flag=False):
        '''
            Update the Observations with a new reading

            Input: 
                State : list(3)
                goal  : list(2)

            * Important:    
                (1) Set the WP0 for Guideline Computation
        '''

        self.Th = self.compute_heading_state(state[2])
        self.Th_e = self.compute_heading_error(state, goal, self.Th)
        self.Xe, self.Ye = self.compute_pos_error(state, goal)
        
        # It's not normalize (Could cause problems)
        self.guideline_dist = self.compute_guideline_dist(state, goal)

        if obst_flag :
            self.obs_dist = self.compute_obs_dist(state, obst_coor)

        # Velocities
        self.Vx = vels[0]
        self.Vy = vels[1]
        self.W = vels[2]

        return self.wrap_output(obst_flag, no_reshape_flag)


    def wrap_output(self, obst_flag=False, no_reshape_flag=False):


        if no_reshape_flag == 0 :
            if obst_flag:
                obs = np.array([self.Xe,
                            self.Ye,
                            self.Th_e,
                            self.Th,
                            self.Vx,
                            self.Vy,
                            self.W,
                            self.guideline_dist,
                            self.obs_dist]).reshape((1, 9))
            else:    
                obs = np.array([self.Xe,
                                self.Ye,
                                self.Th_e,
                                self.Th,
                                self.Vx,
                                self.Vy,
                                self.W,
                                self.guideline_dist]).reshape((1, 8))
        
        else:
            obs = np.array([self.Xe,
                            self.Ye,
                            self.Th_e,
                            self.Th,
                            self.Vx,
                            self.Vy,
                            self.W,
                            self.guideline_dist,
                            self.obs_dist])
        
        return obs
        

    def compute_heading_state(self, heading):
        '''
            Pre-Process the Heading angle
        '''
        angle = self.from360to180(heading)

        return angle

    def from360to180(self, theta):
        '''
            Wrapped the angle if its domain is 0 to 360 
        '''
        if theta > math.pi :
            theta = theta - 2*math.pi
        elif theta < (-math.pi) : 
            theta = theta + 2*math.pi        

        return theta
    
    def compute_heading_error(self, state, goal, theta):

        x_robot = state[0]
        y_robot = state[1]

        ca = x_robot - goal[0]
        co = y_robot - goal[1]

        # angle to the goal
        angle = math.atan(co/ca)
        
        angle = self.angle_correction(angle, co, ca)

        # Heading error
        theta_error = angle - theta

        return theta_error

    def angle_correction(self, angle, co, ca):
        # angle range correction [-180 to 180]
        
        if ca > 0 :
            if co >= 0 :
                angle = angle - math.pi
            
            elif co < 0 :
                angle = math.pi + angle

        return angle
    
    def compute_pos_error(self, state, goal):
        '''
            Position error respect to Goal Coor.
        '''

        xe = goal[0] - state[0]
        ye = goal[1] - state[1]

        # Scaled the error
        xe = xe/self.scale_pos
        ye = ye/self.scale_pos

        return xe, ye
    
    def compute_guideline_dist(self, state, goal):
        
        a = abs( math.sqrt( (goal[0] - state[0] )**2 + (goal[1] - state[1] )**2 ))
        b = abs( math.sqrt( (goal[0] - self.WP0[0])**2 + (goal[1] - self.WP0[1])**2 ))
        c = abs( math.sqrt( (state[0] - self.WP0[0])**2 + (state[1] - self.WP0[1])**2 ))

        if a == 0:
            a = 0.001 # reach the goal

        # Main computation
        if c == 0 :
            relation = 1; # -> theta = 0
        else:
            relation = (a**2 + b**2 - c**2)/(2*a*b)  

        # preventive
        # print("relation = ", relation, a, b, c)
        if abs(relation -1) <= 0.1 :
            relation = 1

        # Angle calculation        
        theta = math.acos( relation )

        # Distance
        dist_goal = a
        distance = dist_goal*math.sin(theta)

        return distance

    def compute_obs_dist(self, state, obs):

        xe = obs[0] - state[0]
        ye = obs[1] - state[1]

        dist = math.sqrt( xe**2 + ye**2 )

        # Scaled Distance
        obs_dist_scaled = dist/self.scale_obs

        # Avoid send the sensor val. directly
        if obs_dist_scaled > 1.25 :
            a = np.random.rand(1) 
            obs_dist = 4 + a.item()
            # obs_dist = 4
        else:
            obs_dist = obs_dist_scaled

        return obs_dist
    
    def locate_obst(self, state, sensor_dist, obst_center_dist=0):

        x_obst_center = state[0] + (sensor_dist+obst_center_dist)*math.cos(state[2])
        y_obst_center = state[1] + (sensor_dist+obst_center_dist)*math.sin(state[2])

        return [x_obst_center, y_obst_center]
