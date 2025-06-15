# import tensorflow as tf
import numpy as np
import math


class policy_manager:
    def __init__(self):
        
        # Agent
        self.id = 0
        self.policy = None                # DRL Model
        self.observations_module = None

        # trajectory
        self.Tr = []        
        self.idx = 0

        self.v = []                       # Direction vector of the current segment


    def initialization(self, id, Tr, policy, observations):        
        self.id = id
        self.Tr = Tr
        self.policy = policy
        self.observations_module = observations

    ## For TensorFlow Model
    # def step(self, state, vels, obst_flag=False, sensor_dist=0, obst_center_dist=0):
    #     '''
    #         state = [robot.x, robot.y, robot.theta]
    #         vels = [robot.Vx, robot.Vy, robot.w]
    #     '''
        
    #     # Goal Coordinate
    #     # self.update_idx(state)
    #     # goal_coor = self.Tr[self.idx+1]

    #     goal_coor = [self.Tr[self.idx+1][0],self.Tr[self.idx+1][1]]   
    #     self.update_idx_reach_goal(state, goal_coor)
    #     goal_coor = self.Tr[self.idx+1]

    #     # Compute Observations
    #     if obst_flag :
    #         obst_coor = self.observations_module.locate_obst(state, sensor_dist, obst_center_dist)
    #     else:
    #         obst_coor = [0, 0]

    #     observations = self.observations_module.compute(state, goal_coor, vels, obst_flag, obst_coor)
    #     obs_input = tf.convert_to_tensor(observations)
    #     action = self.policy.predict(obs_input, verbose=0)

    #     vel_right = action[0][0]
    #     vel_left = action[0][1]

    #     return vel_right, vel_left
    

    def step_np(self, state, vels, obst_flag=False, sensor_dist=0, obst_center_dist=0, no_reshape_flag=True):
        '''
            state = [robot.x, robot.y, robot.theta]
            vels = [robot.Vx, robot.Vy, robot.w]
        '''
        
        # Goal Coordinate
        # self.update_idx(state)
        # goal_coor = self.Tr[self.idx+1]

        goal_coor = [self.Tr[self.idx+1][0],self.Tr[self.idx+1][1]]   
        self.update_idx_reach_goal(state, goal_coor)
        goal_coor = self.Tr[self.idx+1]

        # Compute Observations
        if obst_flag :
            obst_coor = self.observations_module.locate_obst(state, sensor_dist, obst_center_dist)
        else:
            obst_coor = [0, 0]

        obs_input = self.observations_module.compute(state, goal_coor, vels, obst_flag, obst_coor, no_reshape_flag)        
        obs_input = obs_input.astype(np.float32)
        # print(obs_input.dtype)
        action = self.policy.compute(obs_input)
        # print(obs_input.dtype)
        action = action.astype(np.float32)

        vel_right = action[0]
        vel_left = action[1]

        return vel_right, vel_left


    def update_idx_reach_goal(self, state, goal_coor):
        dist_goal = distance(state[0], state[1], goal_coor[0], goal_coor[1])
        
        if dist_goal <= 3 :

            num_points = len(self.Tr)
            if self.idx < num_points-2 :
                self.idx = self.idx + 1
            
            # print("New Goal = ", self.id,  self.Tr[self.idx+1])
            # print()
    

    def update_idx(self, state):
        
        # guideline
        dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]
        dy = self.Tr[self.idx+1][1] - self.Tr[self.idx][1] 


        # State vectors
        self.v = [dx, dy]                       # Direction vector of the current segment
        rx = state[0] - self.Tr[self.idx][0]
        ry = state[1] - self.Tr[self.idx][1]
        r = [rx, ry]

        # Update the Trajectory Segment        
        u = self.dot_mut(self.v, r)/self.dot_mut( self.v, self.v )          # u = v.'*r/(v.'*v) from Matlab

        num_points = len(self.Tr)
        if u > 1 :
            if self.idx < num_points-2 :
                self.idx = self.idx + 1
        


    def dot_mut(self, a, b):
        return a[0]*b[0] + a[1]*b[1] 
    
def distance(xa, ya, xb, yb):
    sum = (xa - xb)**2 + (ya - yb)**2
    
    return math.sqrt(sum)