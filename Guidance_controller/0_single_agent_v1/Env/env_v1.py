import numpy as np
import math

from Env import agents_v1
from Env import env_engine_v1


'''
    Class to define RL enviroment

        1) agents
        2) graphics
        3) RL functions
            3.1) rewards
            3.2) Stop conditions
        4) Step execution


    To Do:
        1) compute_total_reward() : it should be prepare to handle multiple agent rewards (now take just one)

'''

class Environment:
    # def __init__(self, map_dimensions, obstacles, num_agents=1, formation_type=0, mouse_flag=False, reference_path=np.array([])):
    def __init__(self, map_settings, agents_settings, reference_path=np.array([]), training_flag = False):
        
        
        # Init. agents
        self.start_pos_agent = agents_settings['start_pos']
        self.num_agents = agents_settings['num_agents']
        self.formation_type = agents_settings['formation_type']
        self.init_pos_agents = []
        self.agents_obj = []


        # Map
        self.env_map = None
        self.map_dimensions = map_settings['map_dimensions']
        self.img_path = '/home/camilo/Documents/SDU/master/Testing_code/pygame_approach/code_test1/Images' 
        self.map_bkg_path = self.img_path + '/blank_backgroun_0.png'
        self.larger_map_side = 0

        self.num_obstacles = map_settings['num_obs']
        self.obstacles_type = map_settings['type_obs']
        self.seed_rand_obs = map_settings['seed_val_obs']
        self.max_rect_obs_size = map_settings['max_rect_obs_size']
        print('seed val now = ', self.seed_rand_obs)
        self.mouse_flag = map_settings['mouse_flag']

        # Sensor
        self.sensor_range = None
        self.proximity_sensor = None

        # Execution variables
        self.dt = 0
        self.last_time = 0
        self.running_flag = True
        self.pause_sim_flag = False


        # Reference Path
        self.reference_path = reference_path


        # Rewards
        self.reward_ang_error_list = []
        self.reward_distance_list = []
        self.reward_total_list = []
        self.reward_dist_guideline_list = []
        self.reward_orientation_list = []

        self.reward_distance_semiDiscrete_list = []
        self.reward_dist_guideline__semiDiscrete_list = []
        self.reward_orientation_attenuated_list = []
        self.reward_heading_error = []

        self.reward_steps = 0                                    # Penalize for each step running
        self.reward_timeOver = 0                                 # Penalize when stop without reach the goal
        self.stop_by_timeOver = False                            # The flag add a penalty if it True
        self.stop_by_collition = False                           # Check is_alaive() function for the activation
        self.reward_coins = 0
        self.coin_points = np.empty((1,))

        self.goal_reached_flag = False                           # Signal Actived in the goal surroundings to assign an Extra reward 

        # States
        self.state_theta = []                                    # angle between agent and guide line 
        self.state_distance = []                                 # current distance to the goal
        self.state_dist_to_guideline = []                        # distance to the guid line (90 degree angle)
        self.state_orientation = []                              # angle between movement vector to guideline
        self.state_heading = []                                  # Heading angle (kinematics from agent)

        self.factor_norm_dist_guideline = 1                      # self.agent_init_distance_list/factor_norm_dist_guideline to normalize s_ditst_guideline

        # Agent stop
        self.stop_steps = False                                   # Stop iterations if agent is no alive  

        # Record
        self.steps = 0                                            # Iterations counter (env_septs)                                      
        self.max_steps = 100                                      # limit to stop steps
        self.global_iterations = 0
        self.agent_init_distance_list = []                        # Distance to the goal

        # Training 
        self.training_flag = training_flag 

        # Training (Follow WP)
        self.wait_in_wp = 0                                        # Number of times to reach goal to pass to the next one
        self.goal_tolerance = 0.02                                 # Goal zone Margin (around the goal point)


        # Testing Signals
        self.theta_goal_heading_test = []

        

    def initialize(self, env_settings=None):

        # Init. Agents
        self.create_agents()

        # Init Map
        self.init_map()
        self.init_sensors()

        # Axiliar vals.
        self.larger_map_side = self.get_max_map_size()

        # Training settings
        if env_settings != None :

            # Taken from 'map_training_params'
            self.max_steps = env_settings['max_steps']
            self.wait_in_wp = env_settings['wait_in_wp']
            self.goal_tolerance = env_settings['goal_tolerance']


    def init_map(self):
        self.env_map = env_engine_v1.Env_map(self.map_dimensions, self.agents_obj, map_img_path=self.map_bkg_path, mouse_obs_flag=self.mouse_flag)

        self.env_map.max_rect_obs_size = self.max_rect_obs_size
        if self.obstacles_type == 'random':
            self.env_map.random_obstacles(number=self.num_obstacles, seed_val=self.seed_rand_obs) #  (seed 21 / 185 / 285 / 286 ) 88
        elif self.obstacles_type == 'warehouse_0' :
            self.env_map.warehouse_grid(grid_number=0)
        elif self.obstacles_type == 'warehouse_1' :
            self.env_map.warehouse_grid(grid_number=1)

        elif self.obstacles_type == 'center_box' :
            self.env_map.center_box(obst_w = 600, obst_h = 200)
        
        # On obstacles in the map 
        else: 
            self.env_map.random_obstacles(number=0, seed_val=self.seed_rand_obs) 

        # Reference path to be Draw (indicative)
        self.env_map.path_agent = np.copy(self.reference_path)

    def init_sensors(self):
        self.sensor_range = 250, math.radians(40)
        self.proximity_sensor = env_engine_v1.proximity_sensor(self.sensor_range, self.env_map.map)

    def create_agents(self):

        self.agents_init_pos()

        for id, agent_pos in enumerate(self.init_pos_agents):
            self.agents_obj.append( agents_v1.particle(id, agent_pos, self.training_flag) )

    def agents_init_pos(self):
        '''
            Formation type
                0) random
                1) line 
                2) V formation 
        '''

        if self.formation_type == 0:
            self.init_pos_random()

        if self.formation_type == 1:
            self.init_pos_line()

        if self.formation_type == 2:
            self.init_pos_Vformation()

        
    def init_pos_random(self):
        pass
        

    def init_pos_line(self):
        pass

    def init_pos_Vformation(self):
        x0 = self.start_pos_agent[0]  
        y0 = self.start_pos_agent[1]
        dx = 20

        agent_init_pos = (x0, y0)
        self.init_pos_agents.append(agent_init_pos)
        
        for i in range(1, self.num_agents):
            if i%2 : # Odd 
                agent_init_pos = (x0-(i*dx), y0+(i*dx))
            else:
                agent_init_pos = (x0-(i*dx), y0-(i*dx))

            self.init_pos_agents.append(agent_init_pos)

    def reset_env(self):
        self.steps = 0

        # rewards
        self.reward_ang_error_list.clear()
        self.reward_distance_list.clear()
        self.reward_dist_guideline_list.clear()
        self.reward_orientation_list.clear()
        self.reward_total_list.clear()

        self.reward_dist_guideline__semiDiscrete_list.clear()
        self.reward_distance_semiDiscrete_list.clear()
        self.reward_orientation_attenuated_list.clear()
        self.reward_heading_error.clear()

        self.reward_steps = 0
        self.reward_timeOver = 0
        self.stop_by_timeOver = False
        self.stop_by_collition = False
        self.reward_coins = 0
        

        # States
        self.state_theta.clear()                                    # angle between agent and guide line 
        self.state_distance.clear()                                 # current distance to the goal
        self.state_dist_to_guideline.clear()
        self.state_orientation.clear()
        self.state_heading.clear()


        # Agent stop
        self.stop_steps = False  

        ### It could be uncomment (last time)
        # self.env_map.last_time = 0                                # it makes reset dt in env_map.compute_dt() to avoid big space jumps caused by long radients computation



    # def env_step(self, normalize_states = True, training=False, states_reward_flag=True):
    def env_step(self, normalize_states = True, states_reward_flag=True):        
        self.env_map.read_externals(self.agents_obj)                # Read Keyboard commands
        self.running_flag = self.env_map.running
        self.pause_sim_flag = self.env_map.pause_sim_flag
        
        if not(self.env_map.pause_sim_flag):
            self.env_map.compute_dt()                               # Take sim. tame
            self.env_map.map.blit(self.env_map.map_img, (0, 0))            
            
            for agent in self.agents_obj:
                agent.kinematics(self.env_map.dt)
                
                self.env_map.draw_scene(agent)
                agent.collition_flag = self.env_map.collition_flag

                point_cloud = self.proximity_sensor.sense_obstacles(agent.x, agent.y, agent.heading)
                self.env_map.draw_sensor_data(point_cloud)

                # Enviroment states
                if self.training_flag :    
                    self.is_alive(agent)

            if states_reward_flag:
                self.get_output_step(normalize_states)  #*Should be move inside of the for loop 
            
            self.env_map.display_update()
            self.steps = self.steps + 1



    def load_path(self, path_wp, section_type='', points=2):
    
        if section_type == 'div_segments' :
            A = (path_wp[0, 0], path_wp[1, 0])
            B = (path_wp[0, 1],  path_wp[1, 1])
            n = points
            path_wp = self.equidistant_points(A, B, n)
        
        self.reference_path = np.copy(path_wp)        
        self.env_map.path_agent = self.reference_path

        for agent in self.agents_obj:
            # distance_to_goal = agents_v1.distance((agent.x, agent.y), (self.reference_path[0, -1], self.reference_path[1, -1]) )
            distance_to_goal = agents_v1.distance((agent.x, agent.y), (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current]) )
            self.agent_init_distance_list.append(distance_to_goal)
            
            print("Init. distance to the goal ", agent.id, distance_to_goal)
            

    def equidistant_points(self, A, B, n):
        """
            Compute `n` equidistant points along a line segment from A to B.
            
            Parameters:
            A (tuple): Coordinates of the first point (x1, y1)
            B (tuple): Coordinates of the second point (x2, y2)
            n (int): Number of points (including endpoints)
            
            Returns:
            Numpy array: where rows are (x, y) and colmns are all the points
        """
        x1, y1 = A
        x2, y2 = B
        
        # Generate n values of t from 0 to 1
        t_values = np.linspace(0, 1, n)
        
        # Compute x and y coordinates
        x_coords = x1 + t_values * (x2 - x1)
        y_coords = y1 + t_values * (y2 - y1)
        
        
        return np.array([x_coords, y_coords])





    def get_output_step(self, normalize_states = False):
        '''
            Compute States and Rewards
        '''
        
        # Compute angent distance to the goal (wp) 
        # self.state_angl_between(normalize=normalize_states)
        # self.compute_angl_error_reward()

        # Compute angle between agent and guide line to the goal 
        self.compute_state_distance_to_goal(normalize_states=normalize_states)
        self.compute_distance_reward(normalize_states=normalize_states, exp_flag=True)
        self.compute_distance_reward_semiDiscrete(normalize_states=normalize_states, threshold=0.8, attenuation=0.5)
        # print()

        # State agent distance to the guide line
        self.compute_state_dist_guideline(normalize_states=normalize_states)
        self.compute_dist_guideline_reward(normalize_states=normalize_states)
        self.compute_dist_guideline_semiDiscrete_reward(normalize_states=normalize_states, threshold=0.8, attenuation=-0.2) 

        # Heading Angle      
        self.compute_state_heading(norm_flag=False)
        self.compute_reward_heading(exp_flag=True, norm_states=False)

        # State Orientation Angle
        self.compute_state_orientation(normalize_flag=normalize_states)
        self.compute_reward_orientation(normalize_flag=normalize_states)
        self.compute_reward_orientation_attenuated(normalize_flag=normalize_states, threshold=0.05, attenuation=-0.2)

        # Penalization
        self.compute_reward_steps(steps_penalty=0.0001)
        self.compute_timesOver(penalty=50)

        # Discrete Rewards
        init_coor = (self.reference_path[0, 0], self.reference_path[1, 0])
        last_coor = (self.reference_path[0, -1], self.reference_path[1, -1])
        self.compute_reward_coins(init_coor, last_coor, points=6, reward_val=0.01, tol=0.02)
        

        # Compute total reward (Sum all)
        self.compute_total_reward()



    def is_alive(self, agent):
        '''
            The iterations should stop (stop_steps = True)

                1) If the agent collided 
        '''

        # Stopped by Collition
        if agent.collition_flag :
            self.stop_steps = True

            self.stop_by_collition = True
            print("Done by Collistion")

        # Stopped by reach the goal
        _, agent.wp_current, goal_reached = agents_v1.follow_path_wp(agent, self.reference_path, get_angl_flag=False, tolerance= self.goal_tolerance, wait_mode=self.training_flag, wait_steps=self.wait_in_wp)
        if goal_reached :
            self.stop_steps = True
            self.goal_reached_flag = True       # To add extra reward
            agent.collition_flag = True         # To return to init position

            # print("Done by Reaching the Goal --------------------------------------------------- ", agent.wp_current, self.reference_path )
            print("Done by Reaching the Goal --------------------------------------------------- ", agent.wp_current) 
            self.update_wp(agent)
        else:
            self.goal_reached_flag = False  

        # print("Is alive ", self.stop_steps)

        if self.steps >= self.max_steps :
            self.stop_steps = True

            agent.collition_flag = True         # To return to init position (just training)
            self.stop_by_timeOver = True
            print("Done by Steps over")

    
    def update_wp(self, agent):
        '''
            Update Parameters:
                1) The agent init distance to the Goal
        '''
        
        if self.training_flag :
            # agent_x = agent.start_pos[0]
            # agent_y = agent.start_pos[1]
            agent_x = self.reference_path[0, 0]
            agent_y = self.reference_path[1, 0]
        else:
            # Compute the distance for the current segment
            # agent_x = self.reference_path[0, agent.wp_current - 1]
            # agent_y = self.reference_path[1, agent.wp_current - 1]
            agent_x = agent.x
            agent_y = agent.y
        

        distance_to_goal = agents_v1.distance((agent_x, agent_y), (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current]) )
        self.agent_init_distance_list[agent.id] = distance_to_goal
        # print("UPDATE PINT", (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current])  )


    def compute_total_reward(self):
        '''
            Sum. all the reward (max val 1.)
            step reward
        '''

        # Reward Weigths
        # w_ang_error = 0.5
        w_dist_goal = 1
        w_dist_guideline = 1
        w_orientation = 1

        # Extra reward for reach the Goal
        goal_reward = 0.0
        if self.goal_reached_flag :
            goal_reward = 100
        else:
            goal_reward = 0.0

        # Semi-Discrete Total Computation
        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline__semiDiscrete_list[-1][-1] +
                                    #    w_dist_goal* self.reward_distance_semiDiscrete_list[-1][-1]     +
                                        # goal_reward        )
        
        # Mixed Total Computation (Promising)
        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline__semiDiscrete_list[-1][-1] +
        #                               w_dist_goal* self.reward_distance_list[-1][-1]             +
        #                                 goal_reward        )
        
        # Total Reward with Orientation
        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline__semiDiscrete_list[-1][-1] +
        #                                w_dist_goal* self.reward_distance_list[-1][-1]             +
        #                                w_orientation*self.reward_orientation_list[-1][-1]         +
        #                                 goal_reward        )
        
        # Individual Test
        # self.reward_total_list.append( self.reward_coins )
        # self.reward_total_list.append(self.reward_distance_list[-1][-1] + goal_reward + self.reward_timeOver)
        # self.reward_total_list.append(self.reward_distance_list[-1][-1] + self.reward_steps + self.reward_heading_error[-1][-1])
        
        # self.reward_total_list.append(self.reward_distance_list[-1][-1] + self.reward_heading_error[-1][-1])
        # self.reward_total_list.append(self.reward_distance_list[-1][-1] + self.reward_steps + self.reward_heading_error[-1][-1])
        self.reward_total_list.append(self.reward_distance_list[-1][-1] + self.reward_steps + self.reward_heading_error[-1][-1] + self.reward_coins + self.reward_timeOver)

        # self.reward_total_list.append(self.reward_orientation_attenuated_list[-1][-1] + self.reward_steps + self.reward_coins)
        # self.reward_total_list.append(self.reward_orientation_attenuated_list[-1][-1] + self.reward_steps + self.reward_timeOver)
        #self.reward_total_list.append(self.reward_dist_guideline__semiDiscrete_list[-1][-1] + self.reward_steps + self.reward_timeOver)

        # self.reward_total_list.append(self.reward_distance_list[-1][-1] + self.reward_steps + self.reward_dist_guideline__semiDiscrete_list[-1][-1] + self.reward_timeOver)
        
        # Total Reward with Orientation
        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline__semiDiscrete_list[-1][-1] +
        #                                w_dist_goal* self.reward_distance_list[-1][-1]             +
        #                                w_orientation*self.reward_orientation_attenuated_list[-1][-1]         +
        #                                 goal_reward     +
        #                                 self.reward_timeOver )

        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline__semiDiscrete_list[-1][-1] +
        #                                w_dist_goal* self.reward_distance_list[-1][-1]             +
        #                                w_orientation*self.reward_orientation_attenuated_list[-1][-1]         +
        #                                 goal_reward     +
        #                                 self.reward_steps  +
        #                                 self.reward_timeOver )

        # PAST DEFINITIONS
        # self.reward_total_list.append( w_dist_guideline* self.reward_dist_guideline_list[-1][-1] +
        #                               w_dist_goal* self.reward_distance_list[-1][-1]             +
        #                                 goal_reward        )
        
        # w_ang_error* self.reward_ang_error_list[-1]


    def compute_dist_guideline_reward(self, normalize_states=False):

        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]/self.factor_norm_dist_guideline

            if normalize_states:
                current_dist = max_distance*self.state_dist_to_guideline[-1][i]
            else:
                current_dist = self.state_dist_to_guideline[-1][i]

            reward = (max_distance-abs(current_dist))/(max_distance)
            reward_list.append(reward)
            # self.reward_dist_guideline_list.append(reward)
            
            #print('Max. Distance guide line ', max_distance)
            # print('dist. guide line reward = ', reward)
        
        # self.reward_dist_guideline_list.append(reward)
        self.reward_dist_guideline_list.append(reward_list)
        #print('dist. guide line reward = ', self.reward_dist_guideline_list[-1][-1])



    def compute_dist_guideline_semiDiscrete_reward(self, normalize_states=False, threshold=0.9, attenuation=0.1):
        '''
            Attenuation Operation.

                if reward <= threshold :
                    reward = reward*attenuation
        '''

        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]/self.factor_norm_dist_guideline

            if normalize_states:
                current_dist = max_distance*self.state_dist_to_guideline[-1][i]
            else:
                current_dist = self.state_dist_to_guideline[-1][i]

            reward = (max_distance-abs(current_dist))/(max_distance)

            # Semi - Discrete Op.
            # Low reward should be more negative
            if reward <= threshold :
                reward = (1 - reward)*attenuation

            reward_list.append(reward)
                        
        
        # self.reward_dist_guideline_list.append(reward)
        self.reward_dist_guideline__semiDiscrete_list.append(reward_list)
        #print('dist. guide line reward Semi-Discrete = ', self.reward_dist_guideline__semiDiscrete_list[-1][-1])


    def compute_angl_error_reward(self, normilize_states = False):
        '''
            (Discarted)
            Angle between the agent to goal line and guide line

                reward [-1, 1] where (1)  := theta = 0   (degrees)
                               where (-1) := theta = 180 (degrees)

                Note:
                    1) Angle turns to degrees
        '''        
        # reward_val = 1
        ang_zero_y = 0.5                                    # theta > zero_ang_y then negative reward 
        f_zero_y = math.log(ang_zero_y)                     # log_x = b
        max_neg_reward = 6.4                                # -(-math.log(180) + math.log(0.3))
        max_post_reward = 4                                 # maximum value when the reward is positive
        # theta_list = self.state_angl_between()
        # theta_deg = math.degrees(abs(theta_list[0])) 
        
        if normilize_states:
            theta_in = (math.pi/2)*self.state_theta[-1]
        else:
            theta_in = self.state_theta[-1]

        theta_deg = math.degrees(abs(theta_in)) 
        # print('state = ', theta_deg)
        if theta_deg == 0 : 
            log_x = -max_post_reward + f_zero_y
        else:
            log_x = math.log(theta_deg)
        
        # reward = ( -log_x + math.log(ang_zero_y) )
        reward = ( -log_x + f_zero_y )

        # Scale values [1, -1]
        if theta_deg >= ang_zero_y :
            reward = reward/max_neg_reward
        else:
            reward = (reward/max_post_reward)
        
        self.reward_ang_error_list.append( reward )
        # print("reward ang error = ", reward, log_x )



    def compute_distance_reward(self, normalize_states=False, exp_flag=False):
        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]

            if normalize_states:
                # current_dist = max_distance*self.state_distance[i]
                current_dist = max_distance*self.state_distance[-1][i]
            else:
                current_dist = self.state_distance[-1][i]

            reward = (max_distance-current_dist)/(max_distance)

            if exp_flag :
                # reward = math.exp(2*reward) - 1
                reward = math.exp(2*(reward-1) ) - math.exp(-2)

            reward_list.append(reward)
            # self.reward_distance_list.append(reward)
            
            #print('Max. Distance ', max_distance)
            # print('distance reward = ', reward)

        self.reward_distance_list.append(reward_list)
        #print('distance reward = ', self.reward_distance_list[-1][-1])


    def compute_distance_reward_semiDiscrete(self, normalize_states=False, threshold=0.5, attenuation=0.8):
        '''
            Attenuation Ope:

                if reward <= threshold:
                    reward = reward*attenuation
        '''

        reward_list = []

        for i in range(0, len(self.agents_obj)):
            max_distance = self.agent_init_distance_list[i]

            if normalize_states:
                # current_dist = max_distance*self.state_distance[i]
                current_dist = max_distance*self.state_distance[-1][i]
            else:
                current_dist = self.state_distance[-1][i]

            reward = (max_distance-current_dist)/(max_distance)

            # Discrete 
            if reward <= threshold:
                reward = reward*attenuation

            reward_list.append(reward)

        self.reward_distance_semiDiscrete_list.append(reward_list)
        #print('distance reward semi-Discrete = ', self.reward_distance_semiDiscrete_list[-1][-1])


    def compute_state_dist_guideline(self, normalize_states=False):
        '''
            
        '''
        
        distances_goal = self.compute_distance_to_goal(normilize = False)
        theta_list = self.state_angl_between()
        dist_to_guideline_list = []
        sign = 1

        self.factor_norm_dist_guideline = 4

        # larger_map_side = self.get_max_map_size()

        for i, _ in enumerate(self.agents_obj):
            theta = theta_list[i]

            # Just to identify the side respect to the guide-line
            if theta < 0 :
                sign = -1
            else:
                sign = 1

            distance = sign* (distances_goal[i])*math.sin(abs(theta))
            # distance =  (distances_goal[i])*math.sin(abs(theta))

            if normalize_states :
                # distance = distance/self.larger_map_side
                max_dist = self.agent_init_distance_list[i]/self.factor_norm_dist_guideline
                distance = distance/max_dist

            dist_to_guideline_list.append( distance )

        self.state_dist_to_guideline.append(dist_to_guideline_list)

        #print("distance to the guide line ", self.state_dist_to_guideline[-1][-1])


    
    def state_angl_between(self, normalize = False):
        '''
            Theta form 0 to pi
                
                Note: 
                    1) Depends on the side gets (+) or (-) sign
        '''

        # goal_point = (self.reference_path[0, -1], self.reference_path[1, -1])
        # init_point = (self.reference_path[0, -2], self.reference_path[1, -2])
        theta_list = []
        # self.state_theta.clear()

        for agent in self.agents_obj:
            goal_point = (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current])
            init_point = (self.reference_path[0, 0], self.reference_path[1, 0])

            # Cosine Law
            a = agents_v1.distance( goal_point, (agent.x, agent.y) )
            
            # 2-Second computation 
            if a == 0 :
                # privious state to determine the orientation
                # correction based on different triangle definition (agent displacement)
                agent_x = agent.previous_state["x"]
                agent_y = agent.previous_state["y"]
                a = agents_v1.distance( goal_point, (agent_x, agent_y) )
            
            else:
                agent_x = agent.x
                agent_y = agent.y


            b = agents_v1.distance( goal_point, (agent.start_pos[0], agent.start_pos[1]) )
            c = agents_v1.distance( (agent_x, agent_y), (agent.start_pos[0], agent.start_pos[1]) )

            # print("denominator angle ", (a**2 + b**2 - c**2), " / ", 2*a*b)            
            # Main computation
            if c == 0:
                relation = 1 # -> theta = 0
            else:
                relation = (a**2 + b**2 - c**2)/(2*a*b)
            
            # 1-Computation Partially wrong
            # if (abs(relation) < 1.1 ) & (abs(relation) > 0.99 ):
            #     relation = 0.99*(relation/abs(relation))
            

            # Angle calculation
            theta = math.acos( relation )

            # define side (- := up side in the window)
            m = (goal_point[1] - init_point[1])/(goal_point[0] - init_point[0])
            b = goal_point[1] - goal_point[0]*m

            y_line = (agent.x)*m + b
            if y_line > agent.y :
                theta = (-1)*theta

            if normalize :
                theta = theta/(math.pi/2)


            theta_list.append(theta)
            self.state_theta.append(theta)
            # print("Theta = ", math.degrees(theta) )

            
        return theta_list
    
    def compute_state_distance_to_goal(self, normalize_states = False):

        distances = self.compute_distance_to_goal(normilize = normalize_states)

        self.state_distance.append(distances)
        #print('distance to the goal state = ', self.state_distance[-1][-1])



    def compute_distance_to_goal(self, normilize = False):
        # goal_point = (self.reference_path[0, -1], self.reference_path[1, -1])
        # self.state_distance.clear()
        distances = []

        for agent in self.agents_obj:
            goal_point = (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current])
            distance_to_goal = agents_v1.distance((agent.x, agent.y), goal_point)

            if normilize :
                distance_to_goal = distance_to_goal/(self.agent_init_distance_list[agent.id])

            # self.state_distance.append(distance_to_goal)
            distances.append(distance_to_goal)

            # print('distance state = ', distance_to_goal, (agent.x, agent.y))
            # print('distance state = ', distance_to_goal, goal_point, self.agent_init_distance_list[agent.id])
        
        # self.state_distance.append(distances)
        # print('distance state = ', self.state_distance[-1][-1], (agent.x, agent.y))

        return distances


    def compute_state_orientation(self, normalize_flag=False):
        '''
            Compute the angle between the movement direction 
            and the vector from the current position to the goal

                Args:
                    1) normilize_flag : Output form 0 to 1

                return:
                    1) angle: List

        '''
        # goal_point = (self.reference_path[0, -1], self.reference_path[1, -1])
        angles = []

        for agent in self.agents_obj:
            goal_point = (self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current])

            past_x = agent.previous_state["x"]
            past_y = agent.previous_state["y"]

            norm_goal = agents_v1.distance((agent.x, agent.y), goal_point)
            norm_agent = agents_v1.distance((agent.x, agent.y), (past_x, past_y))

            # Avoid Div. by Zero
            if (norm_goal == 0) or (norm_agent == 0) :
                norm_agent = 1
                norm_goal = 1

            # Define the vectors (movement direction) 
            x1 = agent.x - past_x
            y1 = agent.y - past_y
            x2 = goal_point[0] - agent.x
            y2 = goal_point[1] - agent.y

            u_agent = np.array([x1, y1])
            v_goal = np.array([x2, y2])
            
            # Compute angles
            dot_product = np.dot(v_goal, u_agent)
            dot_angle = math.acos(dot_product/(norm_goal*norm_agent))

            cross_product = np.cross(v_goal, u_agent)
            angle_cross = math.asin(cross_product/(norm_goal*norm_agent))

            # cross_angle goes from -90 to 90 degrees (correction)
            if dot_angle > (math.pi/2) :                  
                angle = np.sign(angle_cross)*dot_angle
            else:
                angle = angle_cross

            # Normalization
            if normalize_flag :
                angle = angle/math.pi

            angles.append(angle)

        # print("agent", x1, y1, u_agent)
        # print("target", x2, y2, v_goal)
        # print('dot_angle, angle_cross', dot_angle, angle_cross, angle)
        # print()
        
        self.state_orientation.append(angles)


    def compute_reward_orientation(self, normalize_flag=False):
        '''
            Compute reward based on Oriantetion State
        '''

        reward_list = []

        for i in range(0, len(self.agents_obj)):

            if normalize_flag :
                reward = 1 - abs(self.state_orientation[-1][i])
            else:
                reward = (math.pi - abs(self.state_orientation[-1][i]))/math.pi

            
            reward_list.append(reward)
        
        self.reward_orientation_list.append(reward_list)


    def compute_reward_orientation_attenuated(self, normalize_flag=False, threshold=0.05, attenuation=0.5):
        '''
            Compute reward based on Oriantetion State
            The reward is attenuated regarding a threshold value
        '''

        reward_list = []
        
        for i in range(0, len(self.agents_obj)):
            angle = abs(self.state_orientation[-1][i])
            
            if normalize_flag :
                reward = 1 - angle
            else:
                reward = (math.pi - angle)/math.pi

            # Attenuation
            if angle > (math.pi*threshold) :
                reward = (1 - reward)*attenuation
            
            reward_list.append(reward)
        

        self.reward_orientation_attenuated_list.append(reward_list)


    def compute_reward_steps(self, steps_penalty=0.1):
        '''
            self.reward_steps = self.reward_steps - steps_penalty
        '''
        

        self.reward_steps = self.reward_steps - steps_penalty


    def compute_timesOver(self, penalty=50):
        '''
            If the training stopped by time or collition
        '''
        if self.stop_by_timeOver :
            self.reward_timeOver = -penalty
            self.stop_by_timeOver = False

        # Should be Updated when the Other agents play a role
        if self.stop_by_collition :
            self.reward_timeOver = -penalty
            self.stop_by_collition = False


    def compute_reward_coins(self, A, B, points=8, reward_val=10, tol=0.01):
        '''
            Place rewards along the Guidline 
        '''

        if self.coin_points.shape == (1,) :             
            n = points
            self.coin_points = self.equidistant_points(A, B, n)

        for agent in self.agents_obj:
            agent_x = agent.x
            agent_y = agent.y

            for i in range(1, points): 
                x_wp = self.coin_points[0, i]
                y_wp = self.coin_points[1, i]

                if ( x_wp*(1-tol) <= agent_x ) and ( agent_x <= x_wp*(1+tol) ):
                    if ( y_wp*(1-tol) <= agent_y ) and (  agent_y <=  y_wp*(1+tol) ) :

                        self.reward_coins = reward_val
                    
                    else:
                        self.reward_coins = 0
                    

    def compute_state_heading(self, norm_flag=False):

        heading_angles = []

        for agent in self.agents_obj :
            angle = agent.heading

            # Normalization
            if norm_flag :
                angle = angle/(2*math.pi)
            
            heading_angles.append( angle )
        
        # Update State
        self.state_heading.append( heading_angles )


    def compute_reward_heading(self, exp_flag=True, norm_states=False):
        '''
            Take state computeed with 'compute_state_heading'

                Args:
                    norm_states : If the state are normalized it should be 1

        '''
        reward_list = []        # Agents
        
        for agent in self.agents_obj:
            i = agent.id

            if norm_states :
                heading = (2*math.pi)*self.state_heading[-1][i]
            else:
                heading = self.state_heading[-1][i]

            theta_goal = agents_v1.follow_wp(agent.x, agent.y, self.reference_path[0, agent.wp_current], self.reference_path[1, agent.wp_current])
            theta_error = theta_goal - heading           

            self.theta_goal_heading_test.append( theta_goal )
            if exp_flag :
                # reward = -0.5*math.exp( -40*(theta_error**2) )
                theta_error = theta_error/(math.pi/2)
                reward = 2*math.exp(-2*(theta_error**2)) - 0.5
                # reward = math.exp(-5*(theta_error**2))
            else:
                reward = theta_error

            reward_list.append( reward )
            

        self.reward_heading_error.append( reward_list )


    def apply_actions_left_right(self, action):
        '''
            actions := np.shape(n, 1)
                * n := Number of agents
        '''
        for i, agent in enumerate(self.agents_obj):    
            if action[i, 0] :
                agent.move_right()                    
            else:
                agent.move_left()

    def apply_one_action_left_right(self, action):
        '''
            action := Integer
                * n := Number of agents
        '''
        for i, agent in enumerate(self.agents_obj):    
            if action == 0 :
                agent.move_right()                    

            elif action == 1 :
                agent.move_left()

            elif action == 2 :
                pass


    def get_diagonal_size(self):
        pass

    def get_max_map_size(self):
        x, y = self.map_dimensions

        if x > y : 
            larger = x
        else:
            larger = y


        return larger 

    def visuzalization(self):
        
        print()
        print("---------------------------- Rewards --------------------------------")
        print("Epoch ", self.global_iterations)
        print("Inner iteration ", self.steps)
        print("State Distance - Reward: Linear , Semi_Discrete ")
        print(self.state_distance[-1][-1], " ", self.reward_distance_list[-1][-1], " ", self.reward_dist_guideline_list[-1][-1])
        print()
        print("State Dist. Guidline - Reward: Linear , Semi_Discrete ")
        print(self.state_dist_to_guideline[-1][-1], " ", self.reward_dist_guideline_list[-1][-1], self.reward_dist_guideline__semiDiscrete_list[-1][-1] )
        print()
        print("State Orientation - Reward: Linear , Attenuated ")
        # print(math.degrees(self.state_orientation[-1][-1]), " ", self.reward_orientation_list[-1][-1], " ", self.reward_orientation_attenuated_list[-1][-1])
        print(self.state_orientation[-1][-1], " ", self.reward_orientation_list[-1][-1], " ", self.reward_orientation_attenuated_list[-1][-1])
        print()
        print("Total reward ", self.reward_total_list[-1])
        print("--------------------------------------------------------------------")
        print()


