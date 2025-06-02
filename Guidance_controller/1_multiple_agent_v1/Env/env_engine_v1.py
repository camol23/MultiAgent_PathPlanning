import pygame
import math
import numpy as np
import random
import time

'''
    Based on pygame

        1) Grapical representation 
        2) Sensors

    
    To Do:
        1) initialization function
        2) type of obstacles (now 1 == to random sqaure)
'''


class Env_map:
    def __init__(self, dimentions, robots, map_img_path = None, random_obs_flag = False, num_random_obs = 3, seed_val = None, path_agent=np.array([]), mouse_obs_flag = False):
        pygame.init()

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yel = (255, 255, 0)
        self.steel_blue = (70,130,180)
        self.royal_blue = (65,105,225)
        self.sky_blue = (135,206,235)
        self.gainsboro = (220,220,220)                              # ligth gray
        self.white_smoke = (245,245,245)     
        self.crimson = (220,20,60)     
        self.light_coral = (240,128,128)
        self.medium_aqua_marine = (102,205,170)
        self.plum = (221,160,221)
        self.light_steel_blue = (176,196,222)
        self.slate_gray = (112,128,144)
        self.mint_cream =  	(245,255,250)
        # https://www.rapidtables.com/web/color/RGB_Color.html
                                                                                                # Keep The last one for the goal Triangle
        self.color_list = [self.sky_blue, self.light_coral, self.medium_aqua_marine, self.plum, self.crimson]
        self.color_list_obs = [self.black, self.light_steel_blue, self.slate_gray]

        self.path_agent = path_agent

        # Map
        if map_img_path != None :
            self.map_img = pygame.image.load(map_img_path)

        for robot in robots:
            if (robot.display_type == 1) and (robot.path_img != None) :
                self.robot_img = pygame.image.load(robot.path_img)

        self.background_color = self.white 

        # dimations
        self.width, self.height = dimentions

        # window settings
        pygame.display.set_caption("Obstacle Avoidance")
        self.map = pygame.display.set_mode((self.width, self.height))
        # self.map.fill(self.white)

        if map_img_path != None :
            self.map.blit(self.map_img, (0,0))         

        # robot
        # self.trace_points_list = [robot.start_pos]
        self.resolution_trace = 20                                  # sample each x points
        self.collition_flag = False
        self.collision_color_flag = False

        # Scene 
        self.max_rect_obs_size = 500
        self.random_rect_obs_list = []

        # it was move to the env_v1 init_map() function
        # if random_obs_flag :
        #     self.random_obstacles(num_random_obs, seed_val = seed_val)

        # General
        self.dt = 0                         # current time - old time measure
        self.last_time = 0                  # old time measure
        self.running = True                 # Keep the simulation loop running
        self.pause_sim_flag = False         # Pause simulation pressing (s) key

        # Mouse
        self.rect_mouse = None
        self.pos_mouse = None
        self.mouse_obs_flag =  mouse_obs_flag

        if self.mouse_obs_flag :
            self.rect_mouse = pygame.Rect(0, 0, 25, 25) 
            pygame.mouse.set_visible(False)



    def random_obstacles(self, number, seed_val = None, ):
        ''' 
            the max size of the Obs. is defined in the constructure (map_settings dict) 
        '''        

        # print('seed val = ', seed_val)
        if seed_val != None:
            random.seed(seed_val)

        for i in range(0, number):
            scale_width = random.uniform(0.1, 1)
            scale_height = random.uniform(0.1, 1)

            x_rect = scale_width*self.width
            y_rect = scale_height*self.height
            rect_w = scale_width*self.max_rect_obs_size
            rect_h = scale_height*self.max_rect_obs_size

            rect_w_aux = rect_w + x_rect
            if rect_w_aux > self.width :
                rect_w = self.width - x_rect
            
            rect_h_aux = rect_h + y_rect
            if rect_h_aux > self.height :
                rect_h = self.height - y_rect
            
            self.random_rect_obs_list.append( (int(x_rect), 
                                                int(y_rect), 
                                                int(rect_w), 
                                                int(rect_h)) )
            print("Obst " + str(i) + " = ", self.random_rect_obs_list[i])

    
    def warehouse_grid(self, grid_number = 0):
        '''
            Distribute obstacles with rectangular shape in organize order

                Output: 
                    1) self.random_rect_obs_list =( (int(x_rect), 
                                                     int(y_rect), 
                                                     int(rect_w), 
                                                     int(rect_h)) )

        '''
        
        if grid_number == 0 :
            self.grid0_warehouse()

        elif grid_number == 1 :
            self.grid0_warehouse(num_obst_row = 3, num_obst_colm = 4, distance_between_row  = 40, distance_between_colm = 40, margin_width = 200)

        # DEFAULT (Warehouse Grid)
        else:   
            self.grid0_warehouse()


    def grid0_warehouse(self, num_obst_row = 4, num_obst_colm = 3, distance_between_row  = 50, distance_between_colm = 50, margin_width = 250):
        '''
            Uniform grid with obstacles with the same same and the same distribution in the map
        '''

        print("Warehouse Map selected ")

        # num_obst_row = 4                        # Number obstacles in a row
        # num_obst_colm = 3                       # Number obstacles in a Column
        # distance_between_row  = 50
        # distance_between_colm = 50
        # margin_width = 300                      # Blank distance on the sides

        obst_w = int( (self.width - (2*margin_width) - ((num_obst_row - 1)*distance_between_colm) ) / num_obst_row)
        obst_h = int( (self.height - ((num_obst_colm - 1)*distance_between_row) ) / num_obst_colm)       
        init_x_obst = margin_width
        init_y_obst = 0
        

        for i in range(0, num_obst_row*num_obst_colm):
            row = int(i/num_obst_row)
            colmn = int(i%num_obst_row)

            # print(row, colmn)
            x_pos = colmn*(distance_between_colm + obst_w) + init_x_obst
            y_pos = row*(distance_between_row + obst_h) + init_y_obst

            self.random_rect_obs_list.append( (int(x_pos), 
                                                int(y_pos), 
                                                int(obst_w), 
                                                int(obst_h)) )
            
            print("Obst " + str(i) + " = ", self.random_rect_obs_list[i])


    def center_box(self, obst_w = 600, obst_h = 200):
        '''
            Drawing a rectangle in the middle of the map

            Purposes:
                1) Test conllision functions

        '''      

        x_pos = self.width/2 - (obst_w/2)
        y_pos = self.height/2 - (obst_h/2)

        self.random_rect_obs_list.append( (int(x_pos), 
                                                int(y_pos), 
                                                int(obst_w), 
                                                int(obst_h)) )


    def display_update(self):
        pygame.display.update()


    def draw_scene(self, robot):
        
        # pygame.draw.rect(self.map, self.black, pygame.Rect(0, 0, 1000, 550) )
        for rect_obs in self.random_rect_obs_list:
            pygame.draw.rect(self.map, self.color_list_obs[1], 
                                pygame.Rect( rect_obs[0], rect_obs[1], rect_obs[2], rect_obs[3] ))
                      

        self.draw_mouse_obs()
        self.draw_robot(robot)
        self.draw_path_wp()

    def draw_robot(self, robot):
        
        self.inside_theMap(robot)
        x = robot.x
        y = robot.y
        heading = robot.heading
        type_dis = robot.display_type
        default = False

        # Detect Collision
        # print("Robot position ", (int(x), int(y)), " ", math.degrees(robot.heading), " ", robot.vx  )
        color = self.map.get_at( (int(x), int(y)) )                 
        #if (color[0], color[1], color[2]) != (self.background_color[0], self.background_color[1], self.background_color[2]) :
        self.collision_by_color(color)
        if self.collision_color_flag :
            self.collition_flag = True
        else:
            self.collition_flag = False
        # print("Pixel Color current pos =  ", color, (x, y))
            
        
        # Draw robot
        if (type_dis == 1):
            if (robot.path_img != None) :
                # rotated the robot image and centered in the robot cordinates 
                rotated = pygame.transform.rotozoom(self.robot_img, math.degrees(heading), 1)
                rect = rotated.get_rect(center=(x, y))  
                self.map.blit(rotated, rect)            # overlap the robot image with the map 
            else: 
                default = True

        if (type_dis == 0) or (default ==  True) :
            self.triangle(x, y, heading, 3, robot.id)

        # Draw trace
        for i in range(1, len(robot.trace_path)) :
            # pygame.draw.line(self.map, self.color_list[robot.id], robot.trace_path[i-1], robot.trace_path[i], 4)
            pygame.draw.line(self.map, (self.color_list[robot.id][0]-1, self.color_list[robot.id][1]-1, self.color_list[robot.id][2]-1), robot.trace_path[i-1], robot.trace_path[i], 4)
        
    def inside_theMap(self, robot):
        
        robot.x = max(min( (self.width - 1), robot.x), 0)
        robot.y = max(min( (self.height - 1), robot.y), 0)
         

    def draw_path_wp(self):
        
        if self.path_agent.size == 0 :
            return 

        for i in range(1, self.path_agent.shape[1]):
            pygame.draw.line(self.map, (2, 2, 2), (self.path_agent[0,i-1],self.path_agent[1,i-1]), (self.path_agent[0,i],self.path_agent[1,i]), 2)
            self.triangle(self.path_agent[0,i], self.path_agent[1,i]-8, math.pi/2, side=4, id=-1)


    def draw_mouse_obs(self):
        if self.mouse_obs_flag :
            self.pos_mouse = pygame.mouse.get_pos()
            self.rect_mouse.center = self.pos_mouse
            pygame.draw.rect(self.map, (0,0,0), self.rect_mouse)



    def collision_by_color(self, color):
        obs_flag = []     # collision with an obstacle
        r_flag = []       # collision with a robot
        
        for obs_color in self.color_list_obs :
            obs_flag.append( (color[0], color[1], color[2]) == (obs_color[0], obs_color[1], obs_color[2]) )

        for r_color in self.color_list :
            r_flag.append( (color[0], color[1], color[2]) == (r_color[0], r_color[1], r_color[2]) )

        color_flag_array = np.append( np.array(obs_flag), np.array(r_flag) )
        color_flag_value = np.sum( color_flag_array )
        self.collision_color_flag = (color_flag_value > 0 )

        # obs_flag.clear()
        # r_flag.clear()



    def draw_sensor_data(self, point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map, self.crimson, point, 5, 0)


    def triangle(self, x, y, ang, side, id):        
        x_1 = x - side*np.sin(-ang) - side*np.cos(-ang)
        y_1 = y - side*np.cos( ang) - side*np.sin( ang)
        x_2 = x + side*np.sin(-ang) - side*np.cos(-ang)
        y_2 = y + side*np.cos( ang) - side*np.sin( ang)
        x_3 = x + 2*side*np.cos(-ang)
        y_3 = y + 2*side*np.sin( ang)

        pygame.draw.polygon(self.map, self.color_list[id], [(x_1,y_1),(x_2,y_2),(x_3,y_3)] )


    def compute_dt(self, cte_dt_flag=True, dt_cte=0.007):

        if cte_dt_flag :
            self.dt = dt_cte

        else:
            if self.last_time != 0 :
                self.dt = (pygame.time.get_ticks() - self.last_time)/1000
                self.last_time = pygame.time.get_ticks()
            else:
                self.dt = 0
                self.last_time = pygame.time.get_ticks()


        # print("------------------------------------------ dt = ", self.dt) # rewards debug time = 0.006


    def read_externals(self, agenst_list):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                    
                if event.key == pygame.K_q:
                    self.running = False  

                # Pause the simulation
                if event.key == pygame.K_s:

                    if self.pause_sim_flag :
                      print('Un Paused')
                      self.pause_sim_flag = False
                      self.last_time = pygame.time.get_ticks()  
                    else:
                        print("... PAUSED ... ")
                        self.pause_sim_flag = True

                # if event.key == pygame.K_c:
                #     print("Continuo sim")
                #     flag_wait = False
                #     last_time = pygame.time.get_ticks()

                # Move the agent with the Keyboard
                if event.key == pygame.K_UP:
                    for agent in agenst_list:
                        agent.move_forward()

                if event.key == pygame.K_DOWN:
                    for agent in agenst_list:
                        agent.move_backward()

                if event.key == pygame.K_RIGHT:
                    
                    for agent in agenst_list:
                        agent.heading += math.radians(10)
                    
                if event.key == pygame.K_LEFT:
                    for agent in agenst_list:
                        agent.heading -= math.radians(10)      


##### Sensors

class proximity_sensor():
    def __init__(self, sensor_range, map):

        self.sensor_range = sensor_range
        self.map_width, self.map_height = pygame.display.get_surface().get_size() 
        self.map = map

        self.num_rays = 6

    def sense_obstacles(self, x, y, heading):
        obstacles = []
        x1, y1 = x, y

        self.color_lines = (220,220,220)                                    # gainsboro

        start_angle = heading - self.sensor_range[1]
        finish_angle = heading + self.sensor_range[1]
    
        for angle in np.linspace(start_angle, finish_angle, self.num_rays, False):
            x2 = x1 + self.sensor_range[0] * math.cos(angle)
            y2 = y1 + self.sensor_range[0] * math.sin(angle)

            x2_obs = x2
            y2_obs = y2
            # pygame.draw.line(self.map, (0, 80, 255), (x1, y1), (x2, y2), width=1)

            # Scanning over the line for black pixels
            for i in range(0, 100):
                u = i/100
                # x = int(x2 * u * x1 * (1-u))
                # y = int(y2 * u * y1 * (1-u))
                x = int(u * (x2 - x1) + x1)
                y = int(u * (y2 - y1) + y1)
                # print("Coord sample = ", (x, y), " Map = ", (self.map_width, self.map_height))

                if 0 < x < self.map_width and 0 < y < self.map_height:
                    color = self.map.get_at((x, y))         # pixel
                    #self.map.set_at((x, y), (0, 208, 255))
                    #print("Pixel Color = ", color, " coord = ", (x, y))

                    # Detect the obs. if it is a black pixel
                    if (color[0], color[1], color[2]) == (0, 0, 0) or (color[0], color[1], color[2]) == (176,196,222):
                        #print( "Detected = ", (x, y), " Pixel Color = ", color, 'robot = ', (x1, y1))
                        obstacles.append([x, y])
                        x2_obs = x
                        y2_obs = y
                        break
            
            pygame.draw.line(self.map, self.color_lines, (x1, y1), (x2_obs, y2_obs), width=1)

        
        return obstacles