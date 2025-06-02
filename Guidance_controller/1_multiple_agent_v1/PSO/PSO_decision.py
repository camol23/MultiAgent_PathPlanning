import time
import copy
import numpy as np
import matplotlib.pyplot as plt

# Numpy Permutation (permuted / shuffle)
rng = np.random.default_rng()


'''
    PSO algorith for assign Routes based on distances matrix

'''



class PSO:
    def __init__(self):        

        # Auxiliar
        self.infinity = 10**6

        # PSO Parametera
        self.iter = None            # Max. number of iterations
        self.w = None               # Inertia weight (exploration & explotation)    
        self.Cp = None              # Personal factor        
        self.Cg = None              # Global factor
        self.rp = 0                 # Personal random factor [0, 1]
        self.rg = 0                 # Global random factor [0, 1]
        
        self.Xmin = 0                      
        self.Xmax = 2               # The maximum Value for the Particles
        self.num_particles = None   # Number of Paths (first Method)
        self.resolution = 2         # Numbre of points in the Path


        # PSO Variables
        self.V = None
        self.X = None                                          # Considering the range for the coordinate y_i        
        self.P = None                                          # The best in the Particle
        self.G = None                                          # The best in the Population (Global)
        self.cost_val = None                                   # current Cost value for each particle (1. each path)
        self.p_cost = None                                     # Particle cost val    
        self.g_cost = self.infinity 

        # General
        self.ref_solution = []
        self.dist_matrix = None
        self.output_list = []
        self.output_routes_ids = []

        

    def initialization(self, pso_params, dist_matrix):
        '''
        dist_matrix = (Distances)
                        rows:   Routes
                        colmns: Agents
        '''                 

        # PSO Parametera
        self.iter = pso_params['iterations']                # Max. number of iterations
        self.w = pso_params['w']                            # Inertia weight (exploration & explotation)    
        self.Cp = pso_params['Cp']                          # Personal factor        
        self.Cg = pso_params['Cg']                          # Global factor
        self.rp = 0                                         # Personal random factor [0, 1]
        self.rg = 0                                         # Global random factor [0, 1]
        
        self.num_particles = pso_params['num_particles']    # Number of Paths (first Method)

        self.dist_matrix = copy.deepcopy(dist_matrix)
        self.Xmin = 0                      
        self.Xmax = self.dist_matrix.shape[1] - 1           # agent 0 to agent n
        self.resolution = self.dist_matrix.shape[1]         # Number of agents
        
        # PSO Variables
        self.V = np.zeros((self.num_particles, self.resolution))
        self.X = np.zeros( (self.num_particles, self.resolution) )                                          # Particles x Agent IDs List         
        self.P = np.zeros((self.num_particles, self.resolution))                                            # The best in the Particle
        self.G = np.zeros((self.num_particles, self.resolution))                                            # The best in the Population (Global)
        self.cost_val = np.zeros(self.num_particles)                                                        # current Cost value for each particle (1. each path)
        self.p_cost = np.zeros(self.num_particles)                                                          # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity   

        # Starter Values
        self.reset()

        self.time_ref_path = 0


    def reset(self):
        
        # Get the Best Init Possible for G
        start_time = time.time()
        traditional_path = self.assign_paths(self.dist_matrix)
        self.time_ref_path = time.time() - start_time

        self.ref_solution = copy.deepcopy(traditional_path)
        self.G[:] = traditional_path 
        traditional_path = np.array(traditional_path).reshape(1, len(traditional_path) )
        self.g_cost = self.total_dist(traditional_path).item()
        # print("g cost reset = ", self.g_cost)


        # Initialize The particles
        # the init values are the agent IDs
        ids_list = np.arange(self.resolution)                                                               # Number of agents
        self.X[:] = ids_list
        self.X = rng.permuted(self.X, axis=1)                                                               # Shuffle the IDs in each row/particle
        self.X[0, :] = traditional_path[0, :]
        # print("X in Reset")
        # print(self.X)

        self.V[:,:] = 0        
        self.P[:,:] = 0                                                                                     # The best in the Particle        
        self.cost_val[:] = self.infinity                                                                    # current Cost value for each particle (1. each path)
        self.p_cost[:] = self.infinity 



    def pso_compute(self):
        self.reset()

        for i in range(0, self.iter):

            # Compute Velocity
            r_p = np.random.uniform(0, 1, (self.num_particles, self.resolution))
            r_g = np.random.uniform(0, 1, (self.num_particles, self.resolution))

            self.V = self.w*self.V + \
                    self.Cp*r_p*(self.P - self.X) + \
                    self.Cg*r_g*(self.G - self.X)   

            # Update X 
            self.X = self.X + self.V 
            # print("X")
            # print(self.X)
            self.points_adjustment() 
            self.X = np.int32(self.X)                                                                   # Turns the value to integer (no decimal cordinates)
            self.X = np.float64(self.X)
            # print(self.X)
            # Evaluate Cost value (Updating)
            self.fitness()                                                                              # Compute current Cost value                
            best_cost_mask = self.cost_val < self.p_cost                                                # Compare the current cost against the old value 
            self.p_cost = np.logical_not(best_cost_mask)*self.p_cost + best_cost_mask*self.cost_val     # Update p_cost with the best personal one in the current iteration
            best_cost_mask = best_cost_mask.reshape( (self.X.shape[0], 1) )
            # print(best_cost_mask.shape)
            self.P = np.logical_not(best_cost_mask)*self.P + best_cost_mask*self.X                      # Save old value if current > , and save current when current <
            

            best_index = np.argmin(self.cost_val)                                                       # Take the index of the best particle based on the cost function
            best_current_g_cost = np.min(self.cost_val)
            
            # print("best_current_g_cost ", best_current_g_cost)
            # print("self.g_cost ", self.g_cost)
            # print("self.G ", self.G)

            if best_current_g_cost < self.g_cost :                                                      # If the best current val. is better than the ald global best, then Update 
                self.G[:] = self.X[best_index, :]
                self.g_cost = best_current_g_cost


        # print("Last global best cost value = ", self.g_cost)
        # self.output_path = np.stack( (self.x_fixed, self.G[0, :]) )
        self.output_list = np.int32(self.G[0,:]).tolist()               # pos: Route    | item: Agent_ID
        self.routes2Ids()                                               # pos: Agent_ID | item: Route


    def fitness(self):
        
        # Compute total distance for each particle
        dist_all_particles = self.total_dist(self.X)
        self.X = np.float64(self.X)                         # Inside I turn it to intiger to play as indeces

        # Compute Penalty for Repetition
        penalty_rpt = self.penalty_repeatition()

        # Total Cost
        self.cost_val = penalty_rpt*np.squeeze(dist_all_particles)

        # Vis
        # print("Distances = ", dist_all_particles)
        # print("Penalty = ", penalty_rpt)
        # print("Cost = ", self.cost_val)


    def total_dist(self, id_list):
        '''
            Take the agant IDs and compute the total distance
            for the particle configuration (i.e. route distribution)

            * Turns the Particles matrix from Agent Ids to Agent paths distances
        '''

        id_list = np.int32(id_list)                                                                   # Turns the value to integer (no decimal cordinates)
        # self.X = np.float64(self.X)

        particles_dist = np.zeros_like(id_list)
        for i in range(0, self.dist_matrix.shape[0]) :          # Iter. Through routes 
            
            route_dist_i = self.dist_matrix[i, id_list[:, i]]   # First Route which agent by particle (idx for each angent in the i-th route)
            particles_dist[:, i] = route_dist_i                 # Save the possibilities for the i-th route
            
            # print(route_dist_i.shape, route_dist_i)

        total_cost_particles = np.sum(particles_dist, axis=1)
        # print(total_cost_particles)


        return total_cost_particles


    def penalty_repeatition(self):
        
        penalty_cost = []

        # Read by particle
        for i in range(0, self.X.shape[0] ):

            _, counts = np.unique(self.X[i, :], return_counts=True)
            max_mal = np.max(counts)

            penalty_cost.append( max_mal.item() )

        
        return penalty_cost


    def points_adjustment(self):
        '''
            Keep values in the proper Range and as Intigers
        '''

        self.X = np.round( self.X )
        self.X = np.clip( self.X, a_min=self.Xmin, a_max=self.Xmax)



    def assign_paths(self, dist_matrix):
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
    

    def routes2Ids(self):
        '''
            The Elements in the list are turned in the route number
            and the Position elements are became in the Agent IDs

            output: 
                    [route_1, route_2, ..., route_n ]
        '''

        self.output_routes_ids = [0 for i in range(0, len(self.output_list))]
        for route_i, agent_id in enumerate(self.output_list) :
            
            self.output_routes_ids[agent_id] = route_i

    