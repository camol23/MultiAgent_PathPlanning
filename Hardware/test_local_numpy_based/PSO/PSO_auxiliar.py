import numpy as np
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

