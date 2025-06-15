import math
from utils_fnc import op_funct


class frame_transform:
    def __init__(self):
        
        # Sensor
        self.detection_distance = None

        # Frame are the Transformed Coor.
        # Coor. for the DRL
        # Frame Data 
        self.frame_scale = None
        self.frame_size = None                         # complete size is frame_size*2
        self.r_circ = None                             # real area
        self.circ_margin = None

        self.obst_r_frame = None
        self.obst_r = None

        # DRL map
        self.x0 = None               # Center in real Coordinates
        self.y0 = None   

        self.corners = []

        self.xg_frame = None
        self.yg_frame = None
        self.xr_frame = None
        self.yr_frame = None
        self.xg_sub_frame = None
        self.yg_sub_frame = None

        self.xr_init_frame = None
        self.yr_init_frame = None

        self.sub_goal = None

        # Aux.
        self.tol_sugbgoal = None
        self.init_state = None
        self.goal_coor_init = None

        # Activation
        self.first_detection = False
        self.its_on = False


    def initialization(self, frame_transform_params):
        # Sensor
        self.detection_distance = frame_transform_params['detection_distance']

        # Frame are the Transformed Coor.
        # Coor. for the DRL
        # Frame Data 
        self.frame_scale = frame_transform_params['frame_scale']
        self.frame_size = frame_transform_params['frame_size']                         # complete size is frame_size*2
        self.circ_margin = frame_transform_params['circ_margin']
        self.r_circ = (self.frame_size*self.frame_scale) - self.circ_margin            # real area
        
        self.obst_r_frame = frame_transform_params['obst_r_frame']
        self.obst_r = self.obst_r_frame*self.frame_scale        

        # Aux.
        self.tol_sugbgoal = frame_transform_params['tol_sugbgoal']                      # In real coordinates

    
    def reframe(self, state, goal_coor):
        '''
            Compute the reference frame to 
            apply the Coor. Transformation 
        '''
        
        # Compute obstacle Center
        self.x0 = state[0] + (self.detection_distance + self.obst_r)*math.cos(state[2])
        self.y0 = state[1] + (self.detection_distance + self.obst_r)*math.sin(state[2])
        
        # Compute frame Corners (rela Coor.)
        angle_corner = math.radians(45)

        h_in_frame = math.sqrt( 2*(self.frame_size*self.frame_scale)**2 )
        for i in range(0, 4):
            xi = self.x0 + (h_in_frame)*math.cos(angle_corner)
            yi = self.y0 + (h_in_frame)*math.sin(angle_corner)

            self.corners.append([xi, yi])
            angle_corner = angle_corner + math.radians(90)
        
        # Compute Subgoal
        l_points = [[state[0], state[1]], [goal_coor[0], goal_coor[1]]]
        circle_coor = [self.x0, self.y0]

        point1, point2, delta = op_funct.circle_line_intersection(l_points, circle_coor, self.r_circ)

        ## choose subgoal
        v1 = [ point1[0]-state[0], point1[1]-state[1]]
        v2 = [ point2[0]-state[0], point2[1]-state[1]]
        v_ref = [goal_coor[0] - state[0], goal_coor[1] - state[1]]

        ang_1 = math.atan2(v1[1], v1[0])
        ang_2 = math.atan2(v2[1], v2[0])
        ang_ref = math.atan2(v_ref[1], v_ref[0])

        ang_1 = round(ang_1, 3)
        ang_2 = round(ang_2, 3)
        ang_ref = round(ang_ref, 3)
        print("Re-Frame ")
        print("state ", state, ang_ref)
        print("v1 ", v1, ang_1)
        print("v2 ", v2, ang_2)
        print()
        if ang_1 == ang_2 :
            dist_v1 = op_funct.distance(state[0], state[1], point1[0], point1[1])
            dist_v2 = op_funct.distance(state[0], state[1], point2[0], point2[1])


            if dist_v1 > dist_v2 :
                self.sub_goal = point1
            else:
                self.sub_goal = point2
        else:
            if ang_1 == ang_ref :
                self.sub_goal = point1
            else:
                self.sub_goal = point2

        # Compute Sug.Goal in the new frame
        self.xg_sub_frame, self.yg_sub_frame = op_funct.trans_coor(self.sub_goal, [self.x0, self.y0], self.frame_scale)
        
        self.init_state = state
        self.goal_coor_init = goal_coor
        # self.xr_init_frame, self.yr_init_frame = op_funct.trans_coor([state[0], state[1]], [self.x0, self.y0], self.frame_scale)
        # self.xg_frame, self.yg_frame = op_funct.trans_coor(goal_coor, [self.x0, self.y0], self.frame_scale)

    
    def coor_transformation(self, state, sensor_dist):
        self.xr_frame, self.yr_frame = op_funct.trans_coor([state[0], state[1]], [self.x0, self.y0], self.frame_scale)
        sensor_dist_frame = sensor_dist/self.frame_scale

        return [self.xr_frame, self.yr_frame, state[2]], sensor_dist_frame
    
    def check_activation(self, state, obst_dect_flag, goal_coor):
        '''
            Out if:
                (1) reach sub.goal
                (2) Agent drive out of the boundaries
        '''

        if obst_dect_flag and not(self.first_detection):
            self.its_on = True                      # Activated by sensor
            self.first_detection = True

            self.reframe(state, goal_coor)

        if self.its_on :
            self.first_detection = True             

            # reach the goal            
            dist_sub_goal = op_funct.distance(state[0], state[1], self.sub_goal[0], self.sub_goal[1])            
            if dist_sub_goal <= self.tol_sugbgoal :
                self.its_on = False                         
                # self.first_detection = False             # Temporary (should be deactivated by reachin the goal)
                print("Frame terminated by Reaching the Sub.Goal")
            
            # Out of the Frame
            if len(self.corners) == 0 :
                self.its_on = False                         
                # self.first_detection = False             # Temporary (should be deactivated by reachin the goal)
                print("Frame terminated by Zero Corners")
            
            else:
                x_axis_flag = (state[0] < self.corners[1][0] ) or (state[0] > self.corners[0][0] )
                y_axis_flag = (state[1] < self.corners[2][1] ) or (state[1] > self.corners[1][1] )

                if (x_axis_flag or y_axis_flag) :
                    self.its_on = False                         
                    # self.first_detection = False             # Temporary (should be deactivated by reachin the goal)
                    print("Frame terminated by Out of the Frame")
            
            if self.its_on == 0 :
                print("End DRL frame in state = ", state)




    