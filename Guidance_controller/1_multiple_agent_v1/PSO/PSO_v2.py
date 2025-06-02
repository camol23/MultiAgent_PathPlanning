'''
    PSO for Path planning (customized)
        1) (x, y) are randomized
        2) Cost function penalize the path

        Note: 
            1) num_particles = number of paths

'''

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PSO:
    def __init__(self, map, init_pos, target_pos, pso_params, obs_list):

        self.infinity = 10**6        

        # map properties
        # self.window_w = map.width
        # self.window_h = map.height
        # self.rect_obs_list = map.random_rect_obs_list       # Rectangle shape (x_upper, y_upper, width, height)
        self.window_w, self.window_h = map 

        self.x_init, self.y_init = init_pos
        self.x_target, self.y_target = target_pos

        print(self.x_init, self.y_init)
        print(self.x_target, self.y_target)

        self.obs_rect_list = copy.deepcopy(obs_list)

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
        
        # PSO Variables
        # self.V = np.random.uniform(self.Vmin, self.Vmax, (self.num_particles, self.resolution) )          # Considering the range for the coordinate y_i
        self.V = np.zeros((self.num_particles, self.resolution, 2))
        self.X = np.zeros((self.num_particles, self.resolution, 2))                                         # Considering the range for the coordinate y_i        
        self.P = np.zeros((self.num_particles, self.resolution, 2))                                         # The best in the Particle
        self.G = np.zeros((self.num_particles, self.resolution, 2))                                         # The best in the Population (Global)
        self.cost_val = np.zeros(self.num_particles)                                                        # current Cost value for each particle (1. each path)
        self.p_cost = np.zeros(self.num_particles)                                                          # Particle cost val    
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                         # GLobal cost value

        self.distance = np.zeros(self.num_particles)        # f1 in the fitness function

        # Fixed points (First method)
        # self.diff_xi = np.zeros( (self.num_particles, self.resolution-1) )


    def reset_vals(self):

        self.X[:,:, 0] = np.random.uniform(0, self.window_w, (self.num_particles, self.resolution) )  # Considering the range for the coordinate x_i        
        self.X[:,:, 1] = np.random.uniform(0, self.window_h, (self.num_particles, self.resolution) )  # Considering the range for the coordinate y_i        
        self.V[:,:,:] = 0        
        self.P[:,:,:] = 0                                                                                     # The best in the Particle
        self.G[:,:,:] = 0                                                                                     # The best in the Population (Global)
        self.cost_val[:] = self.infinity                                                                      # current Cost value for each particle (1. each path)
        self.p_cost[:] = self.infinity 
        self.g_cost = self.infinity                                                                           # GLobal cost value

        self.X[:, 0, 0] = self.x_init
        self.X[:, 0, 1] = self.y_init                                                                         # Agent init. position (x_init, y_init)

        self.distance[:] = 0

        self.convert_rect_coord()
        self.Vmax = self.larger_sideMap()



    def convert_rect_coord(self):
        '''
            list = (x_down_left, y_down_left, x_rigth, y_up)
        '''
        # In pygame the axis increase going down in the screen 
        
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][0] + self.obs_rect_list[i][2] 
            rect_h = self.obs_rect_list[i][1] + self.obs_rect_list[i][3] 
            self.obs_rect_list[i] = (self.obs_rect_list[i][0], self.obs_rect_list[i][1], rect_w, rect_h)
        


    def validate_points(self):
        '''
            Identify if the point lands inside of an obstacle, 
            and generate a random replacement around of that obstacle
        '''

        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision = (self.obs_rect_list[i][1] < self.X[:,:,1]) & (self.X[:,:,1] < self.obs_rect_list[i][3])
            
            # x_left < x_fix < x_rigth
            mask_columns = (self.obs_rect_list[i][0] < self.X[:,:,0]) & (self.X[:,:,0] < self.obs_rect_list[i][2])
            mask_collision = mask_collision & mask_columns

            # Then replace the Invalid points
            h_obst = abs(self.obs_rect_list[i][3] - self.obs_rect_list[i][1])
            w_obst = abs(self.obs_rect_list[i][2] - self.obs_rect_list[i][0])
            if w_obst > w_obst:
                larger_side = w_obst
            else: 
                larger_side = h_obst

            X_displaced = self.X + np.random.uniform(-larger_side, larger_side, (self.num_particles, self.resolution, 2))
            self.X[:,:, 0] =  np.logical_not(mask_collision)*self.X[:,:, 0] + mask_collision*X_displaced[:,:, 0]
            self.X[:,:, 1] =  np.logical_not(mask_collision)*self.X[:,:, 1] + mask_collision*X_displaced[:,:, 1]
            

    def collision_inside_obs(self):
        '''
            Detect if the y coordinate of each point is inside of an obstacle
        '''
        mask_collision = np.zeros((self.num_particles, self.resolution), dtype=bool)

        for i in range(0, len(self.obs_rect_list)):
            #  y_botton < X > y_up  (X.shape(num_particles, num_points))
            mask_collision_i = (self.obs_rect_list[i][1] < self.X[:,:,1]) & (self.X[:,:,1] < self.obs_rect_list[i][3])
            
            # x_left < x_fix < x_rigth
            mask_columns = (self.obs_rect_list[i][0] < self.X[:,:,0]) & (self.X[:,:,0] < self.obs_rect_list[i][2])
            mask_collision_i = mask_collision_i & mask_columns

            mask_collision = mask_collision_i | mask_collision


        return mask_collision   # (particles, resolution)
    

    def collision_rect(self):
        '''
            
           d ___ c
            |   |
            |___|
            a    b

        '''
        diff_xi = self.X[:, 1:, 0] - self.X[:, :-1, 0]                  # (particles, resolution-1)
        diff_yi = self.X[:, 1:, 1] - self.X[:, :-1, 1]                  # (particles, resolution-1)

        ones_matrix = np.ones_like(diff_xi)
        mask_div0 = diff_xi == 0
        diff_xi = np.logical_not(mask_div0)*diff_xi + mask_div0*(ones_matrix*1e-6)
        m_i = diff_yi / diff_xi                                         # (particles, resolution-1)

        mask_div0 = (m_i == 0)
        m_i = np.logical_not(mask_div0)*m_i + mask_div0*(ones_matrix*1e-6)

        b_i = self.X[:, 1:, 1] - (self.X[:, 1:, 0]*m_i)                 # (particles, resolution-1)

        mask_ab = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_bc = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_cd = np.zeros((self.num_particles, self.resolution-1), dtype=bool)
        mask_da = np.zeros((self.num_particles, self.resolution-1), dtype=bool)

        for i in range(0, len(self.obs_rect_list)):
            
            # print("Obstacle " + str(i) + " ", self.obs_rect_list[i])
            # print("X = ", self.X)
            # Horizontal segments (ab) and (cd) 
            sign_mask = self.X[:, 1:, 0] < self.X[:, :-1, 0]                                        # The order of the segment points matters
            sign_mask = sign_mask*(-1)

            x_i = (self.obs_rect_list[i][1] - b_i)/m_i                                                                          # (particles, resolution-1) - x over the line from Obst y
            mask_ab_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
            mask_in_line = (sign_mask*self.X[:, :-1, 0] <= sign_mask*x_i) & (sign_mask*x_i <= sign_mask*self.X[:, 1:,0])          # x belongs to the path segment?
            mask_ab_i = mask_ab_i & mask_in_line

            x_i = (self.obs_rect_list[i][3] - b_i)/m_i                                                                            # (particles, resolution-1)
            mask_cd_i = (self.obs_rect_list[i][0] <= x_i) & (x_i <= self.obs_rect_list[i][2])                                     # x overlaps the rectangle segment?
            mask_in_line = (sign_mask*self.X[:, :-1, 0] <= sign_mask*x_i) & (sign_mask*x_i <= sign_mask*self.X[:, 1:,0])          # x belongs to the path segment?
            # print("sign_mask = ", sign_mask)
            # print("x_i = ", x_i, self.X[:, :-1, 0], self.X[:, 1:, 0] )
            # print("mask_in_line ", mask_in_line, " mask_cd_i ", mask_cd_i)
            # print()
            mask_cd_i = mask_cd_i & mask_in_line

            # Vertical segments (bc) and (da) 
            sign_mask = self.X[:, 1:, 1] < self.X[:, :-1, 1]                                                                    # The order of the segment points matters
            sign_mask = sign_mask*(-1)

            y_i = b_i + self.obs_rect_list[i][2]*m_i                                                                            # (particles, resolution-1)
            mask_bc_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])                                     # x overlaps the rectangle segment?    
            mask_in_line = (sign_mask*self.X[:, :-1, 1] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*self.X[:, 1:, 1])         # x belongs to the path segment?
            mask_bc_i = mask_bc_i & mask_in_line

            y_i = b_i + self.obs_rect_list[i][0]*m_i                                                # (particles, resolution-1)
            mask_da_i = (self.obs_rect_list[i][1] <= y_i) & (y_i <= self.obs_rect_list[i][3])       # x overlaps the rectangle segment?     

            mask_in_line = (sign_mask*self.X[:, :-1, 1] <= sign_mask*y_i) & (sign_mask*y_i <= sign_mask*self.X[:, 1:, 1])     # x belongs to the path segment?
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
        '''
            It is not updated, then use fitness_v2()
        '''
        # Shortest Path
        diff_yi = self.X[:, 1:] - self.X[:, :-1]
        diff_coord = np.stack( (self.diff_xi, diff_yi), axis=2 )
        norm_points = np.linalg.norm( diff_coord, axis=2 )

        self.distance = np.sum(norm_points, axis=1)
        self.cost_val = np.sum(norm_points, axis=1)


    def fitness_v2(self):
        '''
            It is included the collision as a penalty factor
            (In process)
        '''

        # Shortest Path
        diff_xi = self.X[:, 1:, 0] - self.X[:, :-1, 0]                  # (particles, resolution-1)
        diff_yi = self.X[:, 1:, 1] - self.X[:, :-1, 1]                  # (particles, resolution-1)
        diff_coord = np.stack( (diff_xi, diff_yi), axis=2 )             # (particles, resolution-1, (x,y))
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
                (In process)
            '''

            # Shortest Path
            diff_xi = self.X[:, 1:, 0] - self.X[:, :-1, 0]                  # (particles, resolution-1)
            diff_yi = self.X[:, 1:, 1] - self.X[:, :-1, 1]                  # (particles, resolution-1)
            diff_coord = np.stack( (diff_xi, diff_yi), axis=2 )             # (particles, resolution-1, (x,y))
            norm_points = np.linalg.norm( diff_coord, axis=2 )              # (particles, resolution-1 )

            self.distance = np.sum(norm_points, axis=1)                     # (particles)

            # Apply penalty to the cost val.
            mask_collision = self.collision_inside_obs()                    # Matrix with one values where a collision is detected
            mask_collision = np.sum(mask_collision, axis=1) > 0             # shape(particles,)
            mask_collision = mask_collision*2

            mask_collision_rect = self.collision_rect()                          # Matrix with one values where a collision is detected
            # print("Intersection Matrix = ", mask_collision_rect.shape," - ", mask_collision_rect)
            mask_collision_rect = np.sum(mask_collision_rect, axis=1) > 0        # shape(particles,)
            mask_collision_rect = mask_collision_rect*8

            # When collided the distance is scaled to discard the path 
            # cost_rect_clollision = np.logical_not(mask_collision_rect)*self.distance + mask_collision_rect*self.distance
            # self.cost_val = np.logical_not(mask_collision)*self.distance + mask_collision*self.distance + cost_rect_clollision

            cost_rect_clollision =  mask_collision_rect*self.distance
            self.cost_val =   mask_collision*self.distance + cost_rect_clollision + self.distance

    def pso_compute(self):
        self.reset_vals()

        for i in range(0, self.iter):
            # self.validate_points()

            # Compute Velocity (could be randmize just for x or y)
            r_p = np.random.uniform(0, 1, (self.num_particles, self.resolution, 2))
            r_g = np.random.uniform(0, 1, (self.num_particles, self.resolution, 2))

            self.V = self.w*self.V + \
                    self.Cp*r_p*(self.P - self.X) + \
                    self.Cg*r_g*(self.G - self.X)   

            # Update X 
            self.X = self.X + self.V 

            self.validate_points()
            self.X = np.int32(self.X)                                                                   # Turns the value to integer (no decimal cordinates)
            self.X = np.float64(self.X)
            self.X[:,:,0] = np.clip(self.X[:,:,0], 0, self.window_w)
            self.X[:,:,1] = np.clip(self.X[:,:,1], 0, self.window_h)

            # Fix target and init position in the paths
            self.X[:, -1, 0] = self.x_target
            self.X[:, 0, 0] = self.x_init
            self.X[:, -1, 1] = self.y_target
            self.X[:, 0, 1] = self.y_init

            # Evaluate Cost value (Updating)
            # self.fitness()                                                                              # Compute current Cost value
            # self.fitness_v2()      
            self.fitness_v3()        
            best_cost_mask = self.cost_val < self.p_cost                                                  # Compare the current cost against the old value 
            self.p_cost = np.logical_not(best_cost_mask)*self.p_cost + best_cost_mask*self.cost_val
            best_cost_mask = best_cost_mask.reshape( (self.X.shape[0], 1) )
            # print(best_cost_mask.shape)
            self.P[:,:, 0] = np.logical_not(best_cost_mask)*self.P[:,:, 0] + best_cost_mask*self.X[:,:, 0]                      # Save old value if current > , and save current when current <
            self.P[:,:, 1] = np.logical_not(best_cost_mask)*self.P[:,:, 1] + best_cost_mask*self.X[:,:, 1]

            best_index = np.argmin(self.cost_val)                                                       # Take the index of the best particle based on the cost function
            best_current_g_cost = np.min(self.cost_val)
            if best_current_g_cost < self.g_cost :                                                      # If the best current val. is better than the ald global best, then Update 
                self.G[:] = self.X[best_index, :, :]
                self.g_cost = best_current_g_cost


        # print("Last global best cost value = ", self.g_cost)
        self.output_path = np.transpose(self.G[0, :, :]) # transposed for environment method
    


    def visualization(self):
        '''
          It should be updated to PSO_v2 from PSO_v1
        '''
        
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1) 
        ax.plot(self.G[0, :, 0], self.G[0, :, 1], color ='tab:red') 
        ax.scatter(self.G[0, :, 0], self.G[0, :, 1], c='red', alpha=0.5, linewidths=0.5)
                    

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][2]
            rect_w = abs(rect_w - self.obs_rect_list[i][0])

            rect_h = self.obs_rect_list[i][3]
            rect_h = abs(rect_h - self.obs_rect_list[i][1])

            x_botton = self.obs_rect_list[i][0]
            y_botton = self.obs_rect_list[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))
        
        ax.set_title('Global best Path') 
        plt.show() 


    def visualization_all(self):
        
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1) 
        
        for i in range(0, self.num_particles):
            ax.plot(self.P[i, :, 0], self.P[i, :, 1], color ='tab:blue', alpha=0.5, linestyle='dashed') 
            ax.scatter(self.P[i, :, 0], self.P[i, :, 1], c='red', alpha=0.5, linewidths=0.5)
        
        # for i in range(1, ( self.x_fixed.shape[0] )-1 ):
        #     ax.plot( [self.x_fixed[i], self.x_fixed[i]], [0, self.window_h], c ='red', alpha=0.5, linestyle='dashed', linewidth=0.5) 

        ax.plot(self.G[0, :, 0], self.G[0, :, 1], color ='tab:red') 
        ax.scatter(self.G[0, :, 0], self.G[0, :, 1], c='red', alpha=0.5, linewidths=0.5)
        

        # Draw obstacles
        for i in range(0, len(self.obs_rect_list)):
            rect_w = self.obs_rect_list[i][2]
            rect_w = abs(rect_w - self.obs_rect_list[i][0])

            rect_h = self.obs_rect_list[i][3]
            rect_h = abs(rect_h - self.obs_rect_list[i][1])

            x_botton = self.obs_rect_list[i][0]
            y_botton = self.obs_rect_list[i][1]

            ax.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor='black'))


        sorted_cost = np.sort(self.p_cost)
        print("Min. cost val = ", np.min(self.p_cost))
        print("Max. cost val = ", np.max(self.p_cost[0]))

        ax.set_title('All Paths - Shortest '+ str(int(sorted_cost[0])) + " - second = "+ str(int(sorted_cost[1])) ) 
        
        plt.show() 

    def larger_sideMap(self):
        
        if self.window_w > self.window_h :
            larger = self.window_w
        else:
            larger = self.window_h

        return larger


    def visualization_lastAdjustment(self):
        pass