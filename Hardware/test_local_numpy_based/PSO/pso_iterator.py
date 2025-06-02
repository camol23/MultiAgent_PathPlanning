
import copy
import numpy as np
# import matplotlib.pyplot as plt

from PSO import PSO_v1



# It must evaluate each route 
# considering the agent Position

class PSO_wrapper:
    def __init__(self):

        # Algorithm
        # self.PSO = None
        self.pso_params = None
        self.obs_list = None
        self.init_pos = None
        self.map_size = None

        self.safe_margin_obs = 15

        # Path
        self.routes = None                      # Possible destination (it should be choosen one)
        self.obs_rect_list_original = None
        
        # Output
        self.dist_list = []        
        self.paths = []
        # self.choosen_Path = []
    

    def initialization(self, map_size, init_pos, routes, pso_params, obs_list):
        '''
            routes : List of Lists 
                        rows: Each route (target_i list)
                        colms: target coordinates
        '''
        self.routes = routes

        # PSO parameters
        self.init_pos = init_pos
        self.pso_params = pso_params
        self.obs_list = obs_list
        self.map_size = map_size



    def itereration(self, plot_flag=False):
        
        # Iteration for each Target Route
        for target in self.routes :
            
            pso = PSO_v1.PSO(self.map_size, self.init_pos, target, self.pso_params, self.obs_list)
            pso.safe_margin = self.safe_margin_obs
            pso.pso_compute()

            # Store Results
            path_i = np.stack( (pso.last_x_output, pso.last_y_output) )            
            self.paths.append( path_i )

            dist_i = self.dist_from_rows(pso.last_x_output, pso.last_y_output)
            self.dist_list.append( dist_i.item() )


            if plot_flag :
                pso.visualization_sequence()

            self.obs_rect_list_original = pso.obs_rect_list_original
            del pso



    def dist_from_rows(self, x_list, y_list):

        diff = np.stack((x_list[1:] - x_list[:-1], y_list[1:] - y_list[:-1]))
        dist = np.linalg.norm(diff, axis=0)

        return np.sum(dist)
    



def assign_paths(dist_matrix):
    '''
        dist_matrix = (Distances)
                        rows:   Routes
                        colmns: Agents
    '''                 

    # Output
    agenst_list = []

    # Rows iterations
    # by Routes
    num_routes = dist_matrix.shape[0]
    matrix = copy.deepcopy(dist_matrix) 
    for i in range(0, num_routes):
        
        if matrix.shape[1] == 1 :             
            mask = dist_matrix[i, :] == matrix[i, 0]
            agenst_list.append( np.argmax(mask).item() )
            break
        
        # Evaluate Route
        dist_route_i = matrix[i, :]
        idx_min = np.argmin(dist_route_i)

        # take the agent Id
        mask = dist_matrix[i, :] == dist_route_i[idx_min]
        agenst_list.append( np.argmax(mask).item() )

        # Discard the Occupied Agent
        matrix = np.delete(matrix, idx_min, axis=1)


    return agenst_list


def take_chosen_routes():
    pass