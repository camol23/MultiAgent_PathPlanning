#!/usr/bin/env python3

import math


class model_diff_v1:
    def __init__(self):

        # Params.
        self.l_width = 0            # Distance between wheels
        self.r_wheel = 0            # Wheel Radius  

        # States
        self.x = 0                  # Current position
        self.y = 0                  # Current position
        self.theta = 0              # Current Angle
        self.x_k = 0                # Previous position
        self.y_k = 0                # Previous position
        self.theta_k = 0            # Previous Angle

        self.x_init = 0             # Init. position     
        self.y_init = 0             # Init. position     
        self.theta_init = 0         # Init. Angle

        self.vr = 0                 # Linear vel. ritght wheel
        self.vl = 0                 # Linear vel. left wheel
        self.v = 0                  # Linear vel. robot
        self.w = 0                  # Angular vel. robot       

        # Aux.
        self.Ts = 0                 # Sample time


    def initialize(self, robot_params):
        # Params.
        self.l_width = robot_params['l_width']              # Distance between wheels
        self.r_wheel = robot_params['r_width']              # Wheel Radius  

        # States        
        self.x_init = robot_params['x_init']                # Init. position     
        self.y_init = robot_params['y_init']                # Init. position     
        self.theta_init = robot_params['theta_init']        # Init. Angle
        self.x = self.x_init                                # Current position
        self.y = self.y_init                                # Current position
        self.theta = self.theta_init                        # Current Angle
        self.x_k = self.x_init                              # Previous position
        self.y_k = self.y_init                              # Previous position
        self.theta_k = self.theta_init                      # Previous Angle
        
        # Aux.
        self.Ts = robot_params['Ts']                        # Sample time

        # Reset    
        self.vr = 0                 # Linear vel. ritght wheel
        self.vl = 0                 # Linear vel. left wheel
        self.v = 0                  # Linear vel. robot
        self.w = 0                  # Angular vel. robot       

        

    def step(self, vr, vl):

        V = (vr + vl)/2
        W = (vr - vl)/self.l_width

        self.x_k = self.x
        self.y_k = self.y
        self.theta_k = self.theta

        # Discrete Model
        x_kp1 = self.x_k + self.Ts*V*math.cos(self.theta_k)
        y_kp1 = self.y_k + self.Ts*V*math.sin(self.theta_k)
        theta_kp1 = self.theta_k + self.Ts*W

        # Output
        self.x = x_kp1
        self.y = y_kp1
        self.theta = theta_kp1

        # store
        self.v = V
        self.w = W
        self.vr = vr
        self.vl = vl
