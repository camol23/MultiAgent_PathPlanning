'''
    PSO for Path planning (customized)

        Note: 
            1) It depends of pygame fuction to detect line object collition 


        ToDo:   
            *the Init Operations should be placed in a Initialization function
'''

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PSO:
    def __init__(self, map, init_pos, target_pos, pso_params, obs_list, seed=None):

        self.infinity = 10**6        

        # map properties
        # self.window_w = map.width
        # self.window_h = map.height
        # self.rect_obs_list = map.random_rect_obs_list       # Rectangle shape (x_upper, y_upper, width, height)
        self.window_w, self.window_h = map 

        self.x_init, self.y_init = init_pos
        self.x_target, self.y_target = target_pos[-1]

        # print(self.x_init, self.y_init)
        # print(self.x_target, self.y_target)

        self.obs_rect_list = copy.deepcopy(obs_list)
        self.obs_rect_list_original = []

        # PSO Parametera
        self.iter = pso_params['iterations']                # Max. number of iterations
        self.w = pso_params['w']                            # Inertia weight (exploration & explotation)    
        self.Cp = pso_params['Cp']                          # Personal factor        
        self.Cg = pso_params['Cg']                          # Global factor
        self.rp = 0                                         # Personal random factor [0, 1]
        self.rg = 0                                         # Global random factor [0, 1]
        # self.Vmin = pso_params['Vmin']                    # Minimum particle Velocity  (Max. range of movement)
        # self.Vmax = pso_params['Vmax']                    # Maximum particle Velocity
        self.Vmin = 0                      
        self.Vmax = self.window_h                           # The maximum for y_i coordinate 
        self.num_particles = pso_params['num_particles']    # Number of Paths (first Method)
        self.resolution = pso_params['resolution']          # Numbre of points in the Path

        self.output_path = None
        # self.output_path_adjusted = None
        self.last_x_output = 0
        self.last_y_output = 0
        
        # PSO Variables
        # print("self.resolution ", self.resolution)
        self.x_fixed = np.linspace(self.x_init, self.x_target, self.resolution)
        # self.x_fixed = np.float64( np.int32(self.x_fixed) )
        self.idx_fixed_x = []
        self.fixed_subGoals = []
        self.subGoals = copy.deepcopy(target_pos[:-1])

        len_target_pose = len(target_pos)
        if len_target_pose > 1:

            for i in range(0, len_target_pose-1) :
                self.fixed_subGoals.append( target_pos[i][0] )  # Take the x coor.

            # Sort Indexes
            x = np.insert(self.x_fixed, 0, self.fixed_subGoals)
            self.x_fixed = np.sort(x)

            # Take the subGoals location
            for xi in self.fixed_subGoals :
                self.idx_fixed_x.append( np.argwhere(self.x_fixed == xi).item() )

        self.resolution = self.x_fixed.shape[0]
        # print("self.resolution END ", self.resolution)

        # self.V = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )          # Considering the range for the coordinate y_i
        self.V = np.zeros((self.num_particles, self.resolution))
        self.X = np.zeros( (self.num_particles, self.resolution) )                                          # Considering the range for the coordinate y_i        
        self.P = np.zeros((self.num_particles, self.resolution))                                            # The best in the Particle
        self.G = np.zeros((self.num_particles, self.resolution))                                            # The best in the Population (Global)
        self.cost_val = np.zeros(self.num_particles)                                                        # current Cost value for each particle (1. each path)
        self.p_cost = np.zeros(self.num_particles)                                                          # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                                # GLobal cost value

        self.cost_penalty = np.zeros(self.num_particles)    # penalty for collition
        self.distance = np.zeros(self.num_particles)        # f1 in the fitness function

        # Fixed points (First method)
        # self.grid_dist = 50                                 # Distance between vertical lines where are placed the y_i points 
        
        # self.x_fixed = self.x_fixed.reshape( (1, self.resolution) )
        self.diff_xi = np.zeros( (self.num_particles, self.resolution-1) )

        self.x_matrix = np.zeros( (self.num_particles, self.resolution) )
        self.x_matrix[:] = self.x_fixed


        # Adjusments
        self.safe_margin = 15
        # self.safe_margin = 8


        if seed != None :
            self.seed = seed
        
        np.random.seed(seed=10)

        self.already_point_creation_flag = 0


    def reset_vals(self):
        # self.V = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )          # Considering the range for the coordinate y_i
        self.X = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )            # Considering the range for the coordinate y_i        
        self.V[:,:] = 0        
        self.P[:,:] = 0                                                                                     # The best in the Particle
        self.G[:,:] = 0                                                                                     # The best in the Population (Global)
        self.cost_val[:] = self.infinity                                                                           # current Cost value for each particle (1. each path)
        self.p_cost[:] = 0                                                                                  # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                                # GLobal cost value

        self.X[:, 0] = self.y_init                                                                          # Agent init. position (x_init, y_init)
        # width_lines = np.abs(self.x_fixed[0] - self.x_fixed[1])
        # self.diff_xi[:,:] = width_lines
        self.diff_xi[:] = self.x_fixed[1:] - self.x_fixed[:-1]        

        self.cost_penalty[:] = 0 
        self.distance[:] = 0

        self.convert_rect_coord()



    def convert_rect_coord(self):
        '''
            list = (x_down_left, y_down_left, x_rigth, y_up)
        '''
        # In pygame the axis increase going down in the screen 
        
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][0] + self.obs_rect_list[i][2]    # x
            rect_h = self.obs_rect_list[i][1] + self.obs_rect_list[i][3]    # y
            
            self.obs_rect_list_original.append((self.obs_rect_list[i][0], self.obs_rect_list[i][1], rect_w, rect_h)) 

            rect_w = rect_w + self.safe_margin
            rect_h = rect_h + self.safe_margin
            self.obs_rect_list[i] = (self.obs_rect_list[i][0] - self.safe_margin, self.obs_rect_list[i][1] - self.safe_margin, rect_w, rect_h)
        

    def particle_collition(self):
        pass

    def validate_points(self):

        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision = (self.obs_rect_list[i][1] < self.X) & (self.X < self.obs_rect_list[i][3])
            
            # x_bottom < x_fix < x_up
            mask_columns = (self.obs_rect_list[i][0] < self.x_fixed) & (self.x_fixed < self.obs_rect_list[i][2])
            mask_collision = mask_collision & mask_columns

            # Then replaced the Invalid points
            # self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(0, self.Vmax, (self.num_particles, self.resolution))
            rand_desition = np.random.rand()
            rand_desition = np.round(rand_desition)
            if rand_desition:
                self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(self.obs_rect_list[i][3], self.window_h, (self.num_particles, self.resolution))
            else:
                self.X =  np.logical_not(mask_collision)*self.X + mask_collision*np.random.uniform(0, self.obs_rect_list[i][1], (self.num_particles, self.resolution))
            

    def collision_inside_obs(self):
        '''
            Detect if the y coordinate of each point is inside of an obstacle
        '''

        mask_collision = np.zeros_like(self.X)
        
        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision = (self.obs_rect_list[i][1] < self.X) & (self.X < self.obs_rect_list[i][3])
            
            # x_bottom < x_fix < x_up
            mask_columns = (self.obs_rect_list[i][0] < self.x_fixed) & (self.x_fixed < self.obs_rect_list[i][2])
            mask_collision = mask_collision & mask_columns

        return mask_collision
    

    def collision_rect(self):
        '''
            
           d ___ c
            |   |
            |___|
            a    b

        '''
        # print("X Shape = ", self.X.shape, self.x_matrix.shape)
        diff_yi = self.X[:, 1:] - self.X[:, :-1]                  # (particles, resolution-1)

        ones_matrix = np.ones_like(diff_yi)        
        m_i = diff_yi / self.diff_xi                              # (particles, resolution-1)

        mask_div0 = (m_i == 0)
        m_i = np.logical_not(mask_div0)*m_i + mask_div0*(ones_matrix*1e-6)

        b_i = self.X[:, 1:] - (self.x_matrix[:, 1:]*m_i)           # (particles, resolution-1)

        mask_ab = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_bc = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_cd = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_da = np.zeros((self.num_particles, self.resolution-1), dtype=bool)

        for i in range(0, len(self.obs_rect_list)):
            
            # print("Obstacle " + str(i) + " ", self.obs_rect_list[i])
            # print("X = ", self.X)

            # Horizontal segments (ab) and (cd) 
            x_i = (self.obs_rect_list[i][1] - b_i)/m_i                                                                          # (particles, resolution-1) - x over the line from Obst y
            mask_ab_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
            mask_in_line = (self.x_matrix[:, :-1] <= x_i) & (x_i <= self.x_matrix[:, 1:])          # x belongs to the path segment?
            mask_ab_i = mask_ab_i & mask_in_line

            x_i = (self.obs_rect_list[i][3] - b_i)/m_i                                                                            # (particles, resolution-1)
            mask_cd_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
            mask_in_line = (self.x_matrix[:, :-1] <= x_i) & (x_i <= self.x_matrix[:, 1:])          # x belongs to the path segment?
            # print("sign_mask = ", sign_mask)
            # print("x_i = ", x_i, self.X[:, :-1, 0], self.X[:, 1:, 0] )
            # print("mask_in_line ", mask_in_line, " mask_cd_i ", mask_cd_i)
            # print()
            mask_cd_i = mask_cd_i & mask_in_line

            # Vertical segments (bc) and (da) 
            sign_mask = self.X[:, 1:] < self.X[:, :-1]                                                                    # The order of the segment points matters
            sign_mask = sign_mask*(-1)

            y_i = b_i + self.obs_rect_list[i][2]*m_i                                                                            # (particles, resolution-1)
            mask_bc_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])                                     # x overlaps the rectangle segment?    
            mask_in_line = (sign_mask*self.X[:, :-1] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*self.X[:, 1:])         # x belongs to the path segment?
            mask_bc_i = mask_bc_i & mask_in_line

            y_i = b_i + self.obs_rect_list[i][0]*m_i                                                # (particles, resolution-1)
            mask_da_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])       # x overlaps the rectangle segment?     
            mask_in_line = (sign_mask*self.X[:, :-1] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*self.X[:, 1:])     # x belongs to the path segment?
            # print("sign_mask = ", sign_mask)
            # print("y_i = ", y_i, self.X[:, :-1, 1], self.X[:, 1:, 1] )
            # print("mask_in_line ", mask_in_line, " mask_da_i ", mask_da_i)
            mask_da_i = mask_da_i & mask_in_line

            mask_ab = mask_ab | mask_ab_i
            mask_bc = mask_bc | mask_bc_i
            mask_cd = mask_cd | mask_cd_i
            mask_da = mask_da | mask_da_i

        # print("mask_ab", mask_ab)
        # print("mask_bc", mask_bc)
        # print("mask_cd", mask_cd)
        # print("mask_da", mask_da)
        return  (mask_ab | mask_bc | mask_cd | mask_da)

    def fitness(self):

        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )
        norm_points = np.linalg.norm( diff_coord, axis=2 )

        self.distance = np.sum(norm_points, axis=1)
        self.cost_val = np.sum(norm_points, axis=1)

    def fitness_v2(self):
        '''
            It is included the collision as a penalty factor
            (In process), sure?
        '''

        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]                        # (particles, resolution-1)
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )        # (particles, resolution-1, (x,y))
        norm_points = np.linalg.norm( diff_coord, axis=2 )              # (particles, resolution-1 )

        self.distance = np.sum(norm_points, axis=1)                     # (particles)

        # Apply penalty to the cost val.
        mask_collision = self.collision_inside_obs()                    # Matrix with one values where a collision is detected
        mask_collision = np.sum(mask_collision, axis=1) > 0             # shape(particles,)
        mask_collision = mask_collision*2

        # When collided the distance is scaled to discard the path 
        self.cost_val = np.logical_not(mask_collision)*self.distance + mask_collision*self.distance
    
    def fitness_v3(self):
        '''
            It is included the collision as a penalty factor

                1) Point inside of the Obstacle
                2) Segment Intersection with the Obstacle  
        '''

        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]                        # (particles, resolution-1)
        # print("self.diff_xi ", self.diff_xi.shape)
        # print("self.diff_yi ",  diff_yi.shape)
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )        # (particles, resolution-1, (x,y))
        norm_points = np.linalg.norm( diff_coord, axis=2 )              # (particles, resolution-1 )

        self.distance = np.sum(norm_points, axis=1)                     # (particles)

        # Apply penalty to the cost val.
        mask_collision = self.collision_inside_obs()                    # Matrix with one values where a collision is detected
        mask_collision = np.sum(mask_collision, axis=1) > 0             # shape(particles,)
        mask_collision = mask_collision*2

        mask_collision_rect = self.collision_rect()                          # Matrix with one values where a collision is detected
        # print("Intersection Matrix = ", mask_collision_rect.shape," - ", mask_collision_rect)
        mask_collision_rect = np.sum(mask_collision_rect, axis=1) > 0        # shape(particles,)
        mask_collision_rect = mask_collision_rect*4

        # When collided the distance is scaled to discard the path 
        # self.cost_val = np.logical_not(mask_collision)*self.distance + mask_collision*self.distance

        cost_rect_clollision =  mask_collision_rect*self.distance
        self.cost_val =   mask_collision*self.distance + cost_rect_clollision + self.distance


    def pso_compute(self):
        self.reset_vals()

        for i in range(0, self.iter):
            # self.validate_points()

            # Compute Velocity
            r_p = np.random.uniform(0, 1, (self.num_particles, self.resolution))
            r_g = np.random.uniform(0, 1, (self.num_particles, self.resolution))

            self.V = self.w*self.V + \
                    self.Cp*r_p*(self.P - self.X) + \
                    self.Cg*r_g*(self.G - self.X)   

            # Update X 
            self.X = self.X + self.V 

            self.validate_points()
            # self.X = np.int32(self.X)                                                                   # Turns the value to integer (no decimal cordinates)
            self.X = np.float64(self.X)
            self.X = np.clip(self.X, 0, self.window_h)

            self.X[:, -1] = self.y_target
            self.X[:, 0] = self.y_init
            # Fixed SubGoals
            if len(self.idx_fixed_x) >0 :
                for i, x_pos in enumerate(self.idx_fixed_x) :
                    self.X[:, x_pos] = self.subGoals[i][1] 

            # Evaluate Cost value (Updating)
            # self.fitness()                                                                              # Compute current Cost value
            # self.fitness_v2()  
            self.fitness_v3()        
            best_cost_mask = self.cost_val < self.p_cost                                                # Compare the current cost against the old value 
            self.p_cost = np.logical_not(best_cost_mask)*self.p_cost + best_cost_mask*self.cost_val
            best_cost_mask = best_cost_mask.reshape( (self.X.shape[0], 1) )
            # print(best_cost_mask.shape)
            self.P = np.logical_not(best_cost_mask)*self.P + best_cost_mask*self.X                      # Save old value if current > , and save current when current <
            

            best_index = np.argmin(self.cost_val)                                                       # Take the index of the best particle based on the cost function
            best_current_g_cost = np.min(self.cost_val)
            if best_current_g_cost < self.g_cost :                                                      # If the best current val. is better than the ald global best, then Update 
                self.G[:] = self.X[best_index, :]
                self.g_cost = best_current_g_cost


        # print("Last global best cost value = ", self.g_cost)
        self.output_path = np.stack( (self.x_fixed, self.G[0, :]) )

        # self.collision_rect_lastCorrection_v2(self.G[0, :])

        # Uncomment when LastCorrection is Omited
        self.last_x_output = self.x_fixed
        self.last_y_output = self.G[0, :]

    
    def collision_rect_lastCorrection_v2(self, path):
        '''
            Each segment intersection is turned to a parallel segment respect to the obstable
                a) If the collision occurs with the last segment it is created a new point

            Objetive: Applying corrections to the output to avoid Path intersections with obstacles 

                    Input:
                        1) Input_Path : dim(1, resolution) : The (y)s coordinates

           d ___ c
            |   |
            |___|
            a    b

        '''

        input_path = np.copy(path)
        x_path = np.copy(self.x_fixed)
        # print("X Shape = ", input_path.shape, x_path.shape)   

        j = 0
        check_collision_flag = False            # True iif is Detected any Collision
        state_flag = False                      # True if is Checking cantidates
        check_coor = []
        alrigth_set_flag = False
        check_counter = 0

        parallel_collision_flag_h = 0       # Intersection with the 2 horizontal lines
        parallel_collision_flag_v = 0       # Intersection with the 2 vertical lines
        previous_parallel = "h"
        intersection_id = []        

        # Units conversion
        target_x = input_path[-1]
        if target_x <= 5 :
            delta_parallel_move = 0.3
        elif target_x <= 20 :
            delta_parallel_move = 3
        elif target_x > 20 :
            delta_parallel_move = 30  

        safe_counter = 0
        while( j < (input_path.shape[0]-1) ):
        # for t in range(0, 11) :
            
            # Compute Segment Line
            diff_yi = input_path[j+1] - input_path[j]                  
            diff_xi = x_path[j+1] - x_path[j]

            x_div0 = (diff_xi == 0)
            diff_xi = np.logical_not(x_div0)*diff_xi + x_div0*(1*1e-6)
            m_i = diff_yi / diff_xi                            

            mask_div0 = (m_i == 0)
            m_i = np.logical_not(mask_div0)*m_i + mask_div0*(1*1e-6)

            b_i = input_path[j+1] - (x_path[j+1]*m_i)          

            # Evaluate the Segment with each Obst.            
            j_backup = j
            for i in range(0, len(self.obs_rect_list)):
                tol_segment = 1*1e-6 # segment tolerance

                # print()
                # print("Input eval Point =", x_path[j+1], input_path[j+1])

                # Vertical segments (bc) and (da) 
                # sign_mask = input_path[j+1] < input_path[j]                                                                    # The order of the segment points matters
                # sign_mask = sign_mask*(-1)
                sign_mask = input_path[j] <= input_path[j+1]                                                                    # The order of the segment points matters
                if sign_mask == 0 :
                    sign_mask = (-1)
                
                tol_segment = tol_segment*sign_mask

                y_i = b_i + self.obs_rect_list[i][0]*m_i                                                                            # (particles, resolution-1)
                mask_bc_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])                                     # x overlaps the rectangle segment?    
                mask_in_line = (sign_mask*(input_path[j]-tol_segment) <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*(input_path[j+1]+tol_segment))         # x belongs to the path segment?
                mask_bc_i = mask_bc_i & mask_in_line

                if mask_bc_i :
                    # print("mask_bc_i")
                    if check_collision_flag and state_flag :                        
                        j = j - 1
                        break

                    coor, x_path, input_path = self.make_segment_parallel_v2(x_path, input_path, j, inter_x=self.obs_rect_list[i][0], inter_y=y_i, orientation = "vertical")
                    check_coor.append(coor)
                    check_collision_flag = True

                    parallel_collision_flag_v = 1
                    intersection_id.append(1)
                    if alrigth_set_flag == False :
                        j_backup = j_backup - 1    # Check again this segment
                        alrigth_set_flag = True                
                    

                y_i = b_i + self.obs_rect_list[i][2]*m_i                                                # (particles, resolution-1)
                mask_da_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])       # x overlaps the rectangle segment?     
                mask_in_line = (sign_mask*(input_path[j]-tol_segment) <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*(input_path[j+1]+tol_segment))     # x belongs to the path segment?
                mask_da_i = mask_da_i & mask_in_line

                if mask_da_i :
                    # print("mask_da_i")
                    if check_collision_flag and state_flag :
                        j = j - 1
                        break

                    coor, x_path, input_path = self.make_segment_parallel_v2(x_path, input_path, j, inter_x=self.obs_rect_list[i][2], inter_y=y_i, orientation = "vertical")
                    check_coor.append(coor)
                    check_collision_flag = True
                    
                    parallel_collision_flag_v = parallel_collision_flag_v + 1
                    intersection_id.append(2)
                    if alrigth_set_flag == False :
                        j_backup = j_backup - 1    # Check agein this segment
                        alrigth_set_flag = True                        


                # Horizontal segments (ab) and (cd) 
                tol_segment = 1*1e-6 # segment tolerance
                sign_mask = x_path[j] <= x_path[j+1]                                                                    # The order of the segment points matters
                if sign_mask == 0 :
                    sign_mask = (-1)

                tol_segment = tol_segment*sign_mask

                x_i = (self.obs_rect_list[i][1] - b_i)/m_i                                                                          # (particles, resolution-1) - x over the line from Obst y
                mask_ab_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
                mask_in_line = (sign_mask*(x_path[j]-tol_segment) <= sign_mask*x_i) & (sign_mask*x_i <= sign_mask*(x_path[j+1]+tol_segment))          # x belongs to the path segment?
                mask_ab_i = mask_ab_i & mask_in_line

                # print(" tol_segment", tol_segment, "sign_mask", sign_mask)
                # print("x_i x_path[j] x_path[j+1]", x_i, x_path[j], x_path[j+1], x_path[j]-tol_segment, x_path[j+1]+tol_segment)
                # print("mask_ab_i ", mask_ab_i, (self.obs_rect_list[i][0] <= x_i), (x_i <= self.obs_rect_list[i][2]) )
                # print('mask_in_line', mask_in_line)
                # print("Obst x = ", self.obs_rect_list[i][0], self.obs_rect_list[i][2])

                if mask_ab_i :
                    # print("mask_ab_i")
                    if check_collision_flag and state_flag :
                        j = j - 1
                        break

                    coor, x_path, input_path = self.make_segment_parallel_v2(x_path, input_path, j, inter_x=x_i, inter_y=self.obs_rect_list[i][1], orientation = "horizontal")
                    check_coor.append(coor)
                    check_collision_flag = True
                    
                    parallel_collision_flag_h = 1
                    intersection_id.append(3)
                    if alrigth_set_flag == False :
                        j_backup = j_backup - 1    # Check agein this segment
                        alrigth_set_flag = True                        
                    

                x_i = (self.obs_rect_list[i][3] - b_i)/m_i                                                                            # (particles, resolution-1)
                mask_cd_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
                mask_in_line = (sign_mask*(x_path[j]-tol_segment) <= sign_mask*x_i) & (sign_mask*x_i <= sign_mask*(x_path[j+1]+tol_segment))          # x belongs to the path segment?                
                mask_cd_i = mask_cd_i & mask_in_line

                if mask_cd_i :
                    # print("mask_cd_i")
                    if check_collision_flag and state_flag :
                        j = j - 1
                        break

                    coor, x_path, input_path = self.make_segment_parallel_v2(x_path, input_path, j, inter_x=x_i, inter_y=self.obs_rect_list[i][3], orientation = "horizontal")
                    check_coor.append(coor)
                    check_collision_flag = True
                    
                    parallel_collision_flag_h = parallel_collision_flag_h + 1
                    intersection_id.append(4)
                    if alrigth_set_flag == False :
                        j_backup = j_backup - 1    # Check again this segment
                        alrigth_set_flag = True
                    
                
                # Validate Point in Collision evaluation
                if state_flag :
                    check_collision_flag = False
                    # print("Out Clean ...")
                
                # Solve One obst. at a time
                if check_collision_flag :
                    # Restore Index
                    j = j_backup 
                    break
            


            # Check the next Segment
            j = j + 1
            if j > (input_path.shape[0]-1) :
                # j = input_path.shape[0]-2
                break
            

            # print("input_path = ", j, x_path)
            # delta_parallel_move = 30
            # Load possible solutions
            if check_collision_flag :

                #Parallel colision 
                # print("In Collision parallel V = ", parallel_collision_flag_v)
                # print("In Collision parallel H = ", parallel_collision_flag_h)
                if parallel_collision_flag_v == 2 :
                    x_path[j+1] = x_path[j] 
                    input_path[j+1] = input_path[j] + delta_parallel_move
                    
                    previous_parallel = "v"
                    # Should be a reset function
                    state_flag = False                      # True if is Checking cantidates
                    check_coor = []
                    check_counter = 0
                    j = j + 1

                elif parallel_collision_flag_h == 2 :
                    x_path[j+1] = x_path[j] + delta_parallel_move
                    input_path[j+1] = input_path[j] 

                    previous_parallel = "h"
                    # Should be a reset function
                    state_flag = False                      # True if is Checking cantidates
                    check_coor = []
                    check_counter = 0
                    j = j + 1

                else: # Common conllision
                                        
                    
                    if (check_counter < len(check_coor)) :
                        check_coor = self.priority_order_checkCoor(check_coor, intersection_id, previous_parallel)
                        # print()
                        # # print("Condition ", check_counter < len(check_coor),check_counter, len(check_coor))
                        # print("Current Point = ", x_path[j+1], input_path[j+1])
                        # print("Candidates = ", check_coor, check_counter)
                        input_path[j+1] = check_coor[check_counter][1]
                        x_path[j+1] = check_coor[check_counter][0]
                        # print("Next Point = ", x_path[j+1], input_path[j+1])

                        state_flag = True

                        # next candidate
                        check_counter = check_counter + 1
                    else:
                        # Force to take the last Candidate (Collision aceepted)
                        # First Method can't deal with that
                        check_collision_flag = False
                        state_flag = False
                        j = j + 1
                        check_coor = []
                        check_counter = 0
            
            # Reset Collision Validation Vals.
            else:
                state_flag = False                      # True if is Checking cantidates
                check_coor = []
                check_counter = 0
                collision_again_flag = False
            
            # Reset
            alrigth_set_flag = False
            parallel_collision_flag_v = 0
            parallel_collision_flag_h = 0
            intersection_id = []
            self.already_point_creation_flag = 0

            safe_counter = safe_counter + 1
            if safe_counter > 100 :
                break

        # Output
        self.last_x_output = x_path
        self.last_y_output = input_path


    def priority_order_checkCoor(self, check_coor, intersection_id, previous_parallel):

        new_coor = []

        # Priority for 2 and 3
        if len(check_coor) > 1 : 
            if previous_parallel == "v" :
                new_coor.append(check_coor[1]) 
                new_coor.append(check_coor[0]) 

            elif previous_parallel == "h" :
                new_coor = check_coor
        else:
            new_coor = check_coor

        return new_coor


    
    def make_segment_parallel_v2(self, input_x, input_y, idx, inter_x=None, inter_y=None, orientation = ""):
        '''
            Move the second point to turn the segment parallel to the obstacle

        '''
        goals = copy.deepcopy(self.subGoals)
        goals.append((self.x_target, self.y_target))

        if self.already_point_creation_flag == 0 :
            for x_target in goals :
                # print("Create point Goals ", x_target)
                # print("Input = ", input_x[idx+1])
                # print()
                if x_target[0] == input_x[idx+1] :
                    input_x, input_y = self.create_new_point(input_x, input_y, idx, inter_x, inter_y)
                    self.already_point_creation_flag = 1


        if orientation == "horizontal" :
            y = input_y[idx].item()   + 0.01 # Update
            x = input_x[idx+1].item()  # Remains the same

        if orientation == "vertical" :
            x = input_x[idx].item()   + 0.01 # Update
            y = input_y[idx+1].item()  # Remains the same


        return (x, y), input_x, input_y
    

    def collision_rect_lastCorrection(self, path):
        '''
            Each segment intersection is turned to a parallel segment respect to the obstable
                a) If the collision occurs with the last segment it is created a new point

            Objetive: Applying corrections to the output to avoid Path intersections with obstacles 

                    Input:
                        1) Input_Path : dim(1, resolution) : The (y)s coordinates

           d ___ c
            |   |
            |___|
            a    b

        '''

        input_path = np.copy(path)
        x_path = np.copy(self.x_fixed)

        print("X Shape = ", input_path.shape, x_path.shape)   
        # for j in range(0, input_path.shape[0]) :
        safe_counter = 0
        j = 0
        while( j < (input_path.shape[0]-1) ):
        # for t in range(0, 2) :
            
            diff_yi = input_path[j+1] - input_path[j]                  # (particles, resolution-1)
            diff_xi = x_path[j+1] - x_path[j]

            x_div0 = (diff_xi == 0)
            diff_xi = np.logical_not(x_div0)*diff_xi + x_div0*(1*1e-6)
            m_i = diff_yi / diff_xi                            # (particles, resolution-1)

            mask_div0 = (m_i == 0)
            m_i = np.logical_not(mask_div0)*m_i + mask_div0*(1*1e-6)

            b_i = input_path[j+1] - (x_path[j+1]*m_i)           # (particles, resolution-1)

            # mask_ab = np.zeros_like(input_path, dtype=bool)
            # mask_bc = np.zeros_like(input_path, dtype=bool)
            # mask_cd = np.zeros_like(input_path, dtype=bool)
            # mask_da = np.zeros_like(input_path, dtype=bool)

            for i in range(0, len(self.obs_rect_list)):
                
                # print("Obstacle " + str(i) + " ", self.obs_rect_list[i])
                # print("X = ", self.X)                

                # Vertical segments (bc) and (da) 
                sign_mask = input_path[j+1] < input_path[j]                                                                    # The order of the segment points matters
                sign_mask = sign_mask*(-1)

                y_i = b_i + self.obs_rect_list[i][0]*m_i                                                                            # (particles, resolution-1)
                mask_bc_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])                                     # x overlaps the rectangle segment?    
                mask_in_line = (sign_mask*input_path[j] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*input_path[j+1])         # x belongs to the path segment?
                mask_bc_i = mask_bc_i & mask_in_line

                if mask_bc_i :
                    print("y_i = ", y_i, input_path[j], input_path[j+1] )
                    print("x_i = ", x_path[j], x_path[j+1] )
                    print("X_i Obs = ", self.obs_rect_list[i][0])
                    x_path, input_path = self.make_segment_parallel(x_path, input_path, j, inter_x=self.obs_rect_list[i][0], inter_y=y_i, orientation = "vertical")
                    print("y_i new = ", input_path[j], input_path[j+1] )
                    print("x_i new = ", x_path[j], x_path[j+1] )
                    print("BC Acctive")
                    print()                    

                    j = j - 1                                    
                    break

                y_i = b_i + self.obs_rect_list[i][2]*m_i                                                # (particles, resolution-1)
                mask_da_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])       # x overlaps the rectangle segment?     
                mask_in_line = (sign_mask*input_path[j] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*input_path[j+1])     # x belongs to the path segment?
                # print("sign_mask = ", sign_mask)
                # print("y_i = ", y_i, input_path[j], input_path[j+1] )
                # print("mask_in_line ", mask_in_line, " mask_da_i ", mask_da_i)
                mask_da_i = mask_da_i & mask_in_line

                if mask_da_i :
                    print("y_i = ", y_i, input_path[j], input_path[j+1] )
                    print("X_i = ", self.obs_rect_list[i][0])
                    x_path, input_path = self.make_segment_parallel(x_path, input_path, j, inter_x=self.obs_rect_list[i][2], inter_y=y_i, orientation = "vertical")

                    print("y_i new = ", input_path[j], input_path[j+1] )
                    print("ad Acctive")
                    print()
                    j = j - 1
                    break

                # Horizontal segments (ab) and (cd) 
                x_i = (self.obs_rect_list[i][1] - b_i)/m_i                                                                          # (particles, resolution-1) - x over the line from Obst y
                mask_ab_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
                mask_in_line = (x_path[j] <= x_i) & (x_i <= x_path[j+1])          # x belongs to the path segment?
                mask_ab_i = mask_ab_i & mask_in_line

                if mask_ab_i :
                    x_path, input_path = self.make_segment_parallel(x_path, input_path, j, inter_x=x_i, inter_y=self.obs_rect_list[i][1], orientation = "horizontal")
                    j = j - 1
                    break

                x_i = (self.obs_rect_list[i][3] - b_i)/m_i                                                                            # (particles, resolution-1)
                mask_cd_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
                mask_in_line = (x_path[j] <= x_i) & (x_i <= x_path[j+1])          # x belongs to the path segment?
                # print("x_i = ", x_i, x_path[j], x_path[j+1] )
                # print("mask_in_line ", mask_in_line, " mask_cd_i ", mask_cd_i)
                # print()
                mask_cd_i = mask_cd_i & mask_in_line

                if mask_cd_i :
                    x_path, input_path = self.make_segment_parallel(x_path, input_path, j, inter_x=x_i, inter_y=self.obs_rect_list[i][3], orientation = "horizontal")
                    j = j - 1
                    break


                # print("J inside = ", j)
                # print("mask_ab", mask_ab_i)
                # print("mask_bc", mask_bc_i)
                # print("mask_cd", mask_cd_i)
                # print("mask_da", mask_da_i)
                # print()

            j = j + 1
            # print("j = ", j)
            if j > (input_path.shape[0]-1) :
                j = input_path.shape[0]-2
            
            safe_counter = safe_counter + 1
            if safe_counter > 100 :
                break

        # self.output_path_adjusted = np.stack( (x_path, input_path) )        
        self.last_x_output = x_path
        self.last_y_output = input_path        
        # return  (mask_ab | mask_bc | mask_cd | mask_da)

    
    def make_segment_parallel(self, input_x, input_y, idx, inter_x=None, inter_y=None, orientation = ""):
        '''
            Move the second point to turn the segment parallel to the obstacle

        '''
        delta = 20

        goals = copy.deepcopy(self.subGoals)
        goals.append((self.x_target, self.y_target))
        for x_target in goals :
            if x_target[0] == input_x[idx+1] :
                input_x, input_y = self.create_new_point(input_x, input_y, idx, inter_x, inter_y)

        print("OUT")
        # if idx == (input_y.shape[0]-2):
        #     input_x, input_y = self.create_new_point(input_x, input_y, idx, inter_x, inter_y)

        if orientation == "horizontal" :
            if input_y[idx + 1] > input_y[idx] :
                delta_plus = delta 
            else:
                delta_plus = -delta             

            input_y[idx + 1] = input_y[idx] + delta_plus

        if orientation == "vertical" :
            if input_x[idx + 1] > input_x[idx] :
                delta_plus = delta 
            else:
                delta_plus = -delta             

            input_x[idx + 1] = input_x[idx] + delta_plus

        return input_x, input_y



    def create_new_point(self, array_x, array_y, idx, x_new, y_new):
        '''
            Includes a new point in the array

                (*) If the y coordinate is equal to the target is added a delta value to progress
        '''
        delta = 5
        diff_y = 0

        if array_y[idx+1] == y_new :            
            i = 0
            while diff_y == 0 :
                diff_y = array_y[idx+1] - array_y[idx-i] 
                i = i + 1
            
            diff_y = diff_y/diff_y 
        
        y_new = y_new + diff_y*delta

        # Include Coordinate
        # array_y = np.insert(array_y, idx+1, y_new)
        # array_x = np.insert(array_x, idx+1, x_new)

        array_y = np.insert(array_y, idx+1, array_y[idx+1])
        array_x = np.insert(array_x, idx+1, array_x[idx+1])

        return array_x, array_y


    def visualization(self, fig=None, row=1, colm=1, pos=1):
        
        if fig == None :
            fig = plt.figure() 
            plot_flag = True
        else: 
            plot_flag = False

        ax = fig.add_subplot(row, colm, pos) 
        ax.plot(self.x_fixed, self.G[0, :], color ='tab:blue') 
        ax.scatter(self.x_fixed, self.G[0, :], c='red', alpha=0.5, linewidths=0.5)
        
        # for xi_dot in self.x_fixed:
            # ax.plot( [xi_dot, xi_dot], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
        for i in range(1, ( self.x_fixed.shape[0] )-1 ):
            ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
            

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list_original)):
            rect_w = self.obs_rect_list_original[i][2]
            rect_w = abs(rect_w - self.obs_rect_list_original[i][0])

            rect_h = self.obs_rect_list_original[i][3]
            rect_h = abs(rect_h - self.obs_rect_list_original[i][1])

            x_botton = self.obs_rect_list_original[i][0]
            y_botton = self.obs_rect_list_original[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))
        
        ax.set_title('Global best Path') 

        if plot_flag :
            plt.show() 


    def visualization_all(self, fig=None, row=1, colm=1, pos=1):
        
        if fig == None :
            fig = plt.figure() 
            plot_flag = True
        else: 
            plot_flag = False

        ax = fig.add_subplot(row, colm, pos) 
        
        for i in range(0, self.num_particles):
            ax.plot(self.x_fixed, self.P[i, :], color ='tab:blue', alpha=0.5, linestyle='dashed') 
            ax.scatter(self.x_fixed, self.P[i, :], c='red', alpha=0.5, linewidths=0.5)
        
        for i in range(1, ( self.x_fixed.shape[0] )-1 ):
            ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 

        ax.plot(self.x_fixed, self.G[0, :], color ='tab:red') 
        ax.scatter(self.x_fixed, self.G[0, :], c='red', alpha=0.5, linewidths=0.5)
        

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list_original)):
            rect_w = self.obs_rect_list_original[i][2]
            rect_w = abs(rect_w - self.obs_rect_list_original[i][0])

            rect_h = self.obs_rect_list_original[i][3]
            rect_h = abs(rect_h - self.obs_rect_list_original[i][1])

            x_botton = self.obs_rect_list_original[i][0]
            y_botton = self.obs_rect_list_original[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))


        sorted_cost = np.sort(self.p_cost)
        print("Min. cost val = ", np.min(self.p_cost))
        print("Max. cost val = ", np.max(self.p_cost[0]))

        ax.set_title('All Paths - Shortest '+ str(int(sorted_cost[0])) + " - second = "+ str(int(sorted_cost[1])) ) 
        
        if plot_flag :
            plt.show() 


    def visualization_lastAdjustment(self, fig=None, row=1, colm=1, pos=1):
        
        if fig == None :
            fig = plt.figure() 
            plot_flag = True
        else: 
            plot_flag = False


        ax = fig.add_subplot(row, colm, pos) 
        ax.plot(self.last_x_output, self.last_y_output, color ='tab:blue') 
        ax.scatter(self.last_x_output, self.last_y_output, c='red', alpha=0.5, linewidths=0.5)
        
        # for xi_dot in self.x_fixed:
            # ax.plot( [xi_dot, xi_dot], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
        for i in range(1, ( self.x_fixed.shape[0] )-1 ):
            ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 
            

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list_original)):
            rect_w = self.obs_rect_list_original[i][2]
            rect_w = abs(rect_w - self.obs_rect_list_original[i][0])

            rect_h = self.obs_rect_list_original[i][3]
            rect_h = abs(rect_h - self.obs_rect_list_original[i][1])

            x_botton = self.obs_rect_list_original[i][0]
            y_botton = self.obs_rect_list_original[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))
        
        ax.set_title('Last adjustment - best Path') 


        if plot_flag :
            plt.show() 



    def visualization_sequence(self):
        '''
            Call Three plotting functions 
             to visualize the process 
        '''

        fig = plt.figure()
        self.visualization(fig, 1, 3, 1)
        self.visualization_all(fig, 1, 3, 2)
        self.visualization_lastAdjustment(fig, 1, 3, 3)
        
        plt.show()