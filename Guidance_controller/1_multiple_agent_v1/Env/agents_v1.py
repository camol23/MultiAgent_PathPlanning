import math
import numpy as np
import random


'''
    1) Agent state
    2) Agent's kinematics
        2.1) Agent's Actions
        
    3) Auxiliar function to handle with state manipulation 

'''
 

def distance(point_a, point_b):
    '''
        Point should be a Tuple (x, y)
    '''
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_dif = point_b - point_a

    # print(" distance points = ", point_dif, " ps = ", point_a, " - ", point_b)
    return np.linalg.norm(point_dif)


class particle:
    def __init__(self, id, start_pos, training_flag=False):
        
        self.id = id
        self.m2p = 3779.52              # meters To pixels

        self.start_pos = start_pos
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.heading = 0

        self.v = 0.01*self.m2p          # m/s
        self.a = 0                      # m/(s^2)

        self.vx = self.v * math.cos(self.heading)
        self.vy = self.v * math.sin(self.heading)
        
        self.max_v = 0.02*self.m2p      # magnitud
        self.min_v = 0.01*self.m2p

        self.previous_state = {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "vel": self.v,
            "accel": self.a
        }

        self.trace_path = [self.start_pos]          # Store (x,y) along the robot displacement
        self.resolution_trace = 20                  # Samples
        self.resolution_counter = 0

        # To be defined ...
        self.min_obs_dist = 100
        self.count_down = 0.5           # seconds

        # Vis
        self.display_type = 0           # Has an image (0: triangle)
        self.path_img = None            # Path to the robot image

        # Collision
        self.collition_flag = False     # Position where the ocollition occurs   
        self.wp_current = 1

        self.training_flag = training_flag
        
        # Follow WPs
        self.wait_counter = 0            # don't read the next wp until reach wait_steps (usefull for training mode)

        
    def move_backward(self):
        self.v = - self.min_v

    def move_forward(self):
        self.v = self.min_v

    def move_rotate(self, ang):
        self.heading = ang

    def move_stop(self):
        self.v = 0

    def move_start(self):
        self.v = self.min_v  

    def move_right(self, angl=5):
        self.heading += math.radians(angl)

    def move_left(self, angl=5):
        self.heading -= math.radians(angl)

    def reset_agent(self):
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.heading = 0

        
    
    def kinematics(self, dt):

        if self.collition_flag :
        #     self.x, self.y = self.collition_state 
            self.x = self.previous_state["x"]
            self.y = self.previous_state["y"]
            self.heading = self.previous_state["heading"]
            self.v = self.previous_state["vel"]
            self.a = self.previous_state["accel"]

            if self.training_flag :
                self.reset_agent()
        
        self.store_trace()

        # Kinematics
        # if self.heading > (2*math.pi) or self.heading < (-2*math.pi):
        #     self.heading = 0
        if self.heading > (math.pi) :
            self.heading = math.pi - self.heading 

        elif self.heading < (-math.pi) :
            self.heading = - math.pi - self.heading 

        self.v = self.v + self.a*dt
        
        self.vx = self.v * np.cos(self.heading)
        self.vy = self.v * np.sin(self.heading)
        
        self.x = self.x + self.vx*dt
        self.y = self.y + self.vy*dt


        self.vx = max(min(self.max_v, self.vx), self.min_v)
        self.vy = max(min(self.max_v, self.vy), self.min_v)


    def store_trace(self):
        self.previous_state["x"] = self.x
        self.previous_state["y"] = self.y
        self.previous_state["heading"] = self.heading
        self.previous_state["vel"] = self.v
        self.previous_state["accel"] = self.a

        if self.resolution_counter >= self.resolution_trace :
            self.trace_path.append( (self.x, self.y) )
            self.resolution_counter = self.resolution_trace

            # Avoid memory exhaustion
            if len(self.trace_path) > 5000 :
                del self.trace_path[0]
        
        self.resolution_counter += 1




def follow_wp(x_robot, y_robot, x_wp, y_wp):
    # path_wp := numpy( row 0: x,  row 1: y)) 

    # h = distance((x_robot, y_robot), (x_wp, y_wp))
    # if h == 0:
    #     h = 0.001
    co = y_robot - y_wp 
    ca = x_robot - x_wp

    # angle = math.acos(ca/h)
    angle = math.atan(co/ca)
    
    if ca > 0 :

        if co >= 0 :
            angle = angle - math.pi
        elif co < 0 :
            angle = math.pi + angle

    # print("angle = ", angle)
    return angle

def follow_path_wp(robot, path_wp, get_angl_flag=True, tolerance=0.02, wait_mode=False, wait_steps=0):
    # path_wp := numpy( row 0: x,  row 1: y)) 
    stop_signal = 0
    x_robot = robot.x
    y_robot = robot.y

    idx_wp = robot.wp_current

    
    x_wp = path_wp[0,idx_wp]
    y_wp = path_wp[1,idx_wp]

    x_wp, y_wp = swarm_path_adjustment(robot, x_wp, y_wp)
        

    tol = tolerance
    # if ((x_wp*(1-tol), y_wp*(1-tol)) <= (x_robot, y_robot)) & ((x_robot, y_robot) <= (x_wp*(1+tol), y_wp*(1+tol))):
    if ( x_wp*(1-tol) <= x_robot ) and ( x_robot <= x_wp*(1+tol) ):
        if ( y_wp*(1-tol) <= y_robot ) and (  y_robot <=  y_wp*(1+tol) ) :
            
            if wait_mode and (wait_steps > robot.wait_counter) :
                robot.wait_counter += 1
                print("Wait COunter = ", robot.wait_counter)
            else: 
                idx_wp = idx_wp+1   
                robot.wait_counter = 0
                print("New WP Idex = ", idx_wp)

            stop_signal = 1

            if idx_wp == (path_wp.shape[1]):
                stop_signal = 1
                idx_wp -= 1

        # x_wp = path_wp[0,idx_wp]
        # y_wp = path_wp[1,idx_wp]


    if get_angl_flag :
        angle = follow_wp(x_robot, y_robot, x_wp, y_wp)
    else:
        angle = None

    return angle, idx_wp, stop_signal

def swarm_path_adjustment(robot, x_wp, y_wp):
    id = robot.id
    dx = 20

    if id == 1:
       x_wp = x_wp-dx
       y_wp = y_wp+dx
    
    if id == 2:
       x_wp = x_wp-dx
       y_wp = y_wp-dx

    return x_wp, y_wp




####################### Additional Robots #######################

class wheel_robot(particle):
    def __init__(self, start_pos, width):
        super().__init__(start_pos)
        
        # self.m2p = 3779.52          # meters To pixels
        self.w = width              # robot width (L)

        # self.start_pos = start_pos
        # self.x = start_pos[0]
        # self.y = start_pos[1]
        # self.heading = 0

        self.vl = 0.01*self.m2p         # m/s
        self.vr = 0.01*self.m2p

        # self.max_v = 0.02*self.m2p
        # self.min_v = 0.01*self.m2p

        # self.min_obs_dist = 100
        # self.count_down = 0.5           # seconds

        self.display_type = 1                         # Has an image (1:)
        self.path_img = './Images/robot_1.png'        # Path to the robot image

    
    def avoid_obstacle(self, point_cloud, dt):
        closet_obs = None
        dist = np.inf

        if len(point_cloud) > 1:                    # there are sensor info
            for point in point_cloud:
                distTopoint = distance([self.x, self.y], point)

                if dist > distTopoint:
                    dist = distTopoint
                    closet_obs = (point, dist)

            
            if closet_obs[1] < self.min_obs_dist and self.count_down > 0:
                self.count_down -= dt               # Stop moving back (count_down)
                self.move_backward()
            else:
                self.count_down = 5
                self.move_forward()

        
    def move_backward(self):
        self.vr = - self.min_v
        self.vl = - self.min_v/2

    def move_forward(self):
        self.vr = self.min_v
        self.vl = self.min_v

    
    def kinematics(self, dt):

        self.x += ((self.vl + self.vr)/2) * math.cos(self.heading) * dt
        self.y += ((self.vl + self.vr)/2) * math.sin(self.heading) * dt
        self.heading += (self.vr - self.vl) / self.w * dt

        if self.heading > (2*math.pi) or self.heading < (-2*math.pi):
            self.heading = 0

        self.vr = max(min(self.max_v, self.vr), self.min_v)
        self.vl = max(min(self.max_v, self.vl), self.min_v)