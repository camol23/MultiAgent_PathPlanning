import numpy as np
import math
from PSO import PSO_v1


class pso_map_update:
    def __init__(self):
        
        # PSO
        self.pso_params = None
        self.obs_list = None
        self.init_pos = None
        self.map_size = None
        self.targets = None
        self.safe_margin_obs = 15

        self.robot_state = None
        self.new_track = None

        # Auxiliar
        self.divider = 1                # Divide the output coordinates


    def initialization(self, map_size, state, targets, pso_params, obs_list, idx, route):
        '''
            route  : It's the previous aproach
            target : They are the original Goals
            Idx    : It's the current WP 
        '''
        self.targets = self.validate_goals(targets, route, idx)

        # PSO parameters
        self.init_pos = state[:2]
        self.pso_params = pso_params
        self.obs_list = obs_list
        self.map_size = self.map_size_adjusment(state, map_size)
        self.robot_state = state


    def reset(self):
        pass

    def compute_new_route(self):

        pso = PSO_v1.PSO(self.map_size, self.init_pos, self.targets, self.pso_params, self.obs_list)
        pso.safe_margin = self.safe_margin_obs
        pso.pso_compute()

        # Solution
        path = np.stack( (pso.last_x_output, pso.last_y_output) )            
        self.new_track = [[path[0][i].item()/self.divider, path[1][i].item()/self.divider] for i in range(0, len(path[0]))]



    def validate_goals(self, targets, route, idx):
        '''
            Check if the Mandatory Stops (Goals)
            have been reached previously to be discarted
            in te new solution

            idx : It's the reference index, the agent already passed this one
                and looking for Idx+1
        '''
        
        valid_targets = []
        route_np = np.array(route[:idx+1])    # --> [0 to idx] or [0 to idx+1)
        for goal in targets:
            x_mask = (route_np[:, 0] == goal[0])
            y_mask = (route_np[:, 1] == goal[1])

            point_mask = x_mask*y_mask
            sum_mask = np.sum(point_mask)

            if sum_mask == 0 :
                valid_targets.append(goal)


        return valid_targets
    

    def map_size_adjusment(self, state, map_size):

        new_x = state[0] - 10/self.divider

        return (new_x, map_size[1])


# This one should be only one class with pso_map_update

# default obst example = [(75, 125, 20, 40)]
class obst_persistance:
    def __init__(self):
        self.counter = None
        self.counter_limit = None
        self.activation_flag = None

        self.dist_to_obst = None
        self.detection_margin = None

        self.default_obst_width = None
        self.default_obst_height = None
        self.units_divider = None

        self.unknown_obst_list = None


    def initialization(self, obst_persistance_params):
        
        self.counter = 0
        self.counter_limit = obst_persistance_params['counter_limit']
        self.activation_flag = 1        

        self.units_divider = obst_persistance_params['units_divider']
        self.default_obst_width = obst_persistance_params['default_obst_width']/self.units_divider
        self.default_obst_height = obst_persistance_params['default_obst_height']/self.units_divider

        self.detection_margin = obst_persistance_params['detection_margin']/self.units_divider
        self.unknown_obst_list = []
        


    def add_obst_by_detection(self, state, dist):
        '''
            Add and obstacle in the list
            in the position of the detection

            (*) The obstacle has a default size
        '''

        dist_units = dist/self.units_divider

        if dist_units <= self.detection_margin :
            x_obst_center = state[0] + dist_units*math.cos(state[2])
            y_obst_center = state[1] + dist_units*math.sin(state[2])

            x_obst_lower = x_obst_center #- self.default_obst_width/4  # 4 better   #/2
            y_obst_lower = y_obst_center - self.default_obst_height/2 # 4 better #/2

            new_obst = (x_obst_lower, y_obst_lower, self.default_obst_width, self.default_obst_height)

            self.unknown_obst_list.append( new_obst )


    def detection_counter(self):
        '''
            activation_flag is On when the limit is reached
        '''
        
        self.counter = self.counter + 1
        if self.counter >= self.counter_limit :
            self.activation_flag = 1
            self.counter = 0
        else:
            self.activation_flag = 0

        return self.activation_flag