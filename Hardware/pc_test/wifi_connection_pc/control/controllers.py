#!/usr/bin/env python3

import math



class move_forward:
    def __init__(self, mf_control_params):

        # Gians
        self.kv = mf_control_params['kv']               # Vel. Gain         (sim = 0.8)
        self.kw = mf_control_params['kw']               # Angular Vel. Gain (sim = 5)

        # Goal
        self.xg = mf_control_params['xg']               # Goal x coor.
        self.yg = mf_control_params['yg']               # Goal y coor.

        # Aux. 
        self.l_width = mf_control_params['l_width']    # robot width (0.105)


    def step(self, state, vis=False):
        
        # Theta ref.
        robot_x = state[0]
        robot_y = state[1]
        theta_ref = self.compute_theta_ref(robot_x, robot_y)

        # Compute Error
        x_e = self.xg - robot_x
        y_e = self.yg - robot_y
        theta_e = theta_ref - state[2]

        # Controller
        v = self.kv*math.sqrt( x_e**2 + y_e**2 )
        w = self.kw*theta_e

        # Output
        vr, vl = self.output_format(v, w)


        # Vis.
        if vis :
            print("Vels. Controller (i.e. vr, vl, v, w) = ", vr, vl, v, w)
        
        return vr, vl
    

    def compute_theta_ref(self, robot_x, robot_y):
        '''
            COmpute the desire heading angle considering 
            the current robot Coordinates
        '''
        # Catetos
        co = self.yg - robot_y
        ca = self.xg - robot_x

        # Desire Angle
        theta_ref = math.atan2(co, ca)

        return theta_ref
    
    
    def output_format(self, v, w):

        # Model Equations
        # V = (vr + vl)/2;
        # w = (vr - vl)/l_width;
        #
        # vr = l_width*w + vl
        # V = (l_width*w + vl + vl)/2;

        # Solution
        vl = (2*v - self.l_width*w)/2
        vr = self.l_width*w + vl

        return vr, vl
    


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class path_follower:
    def __init__(self, pf_control_params):
        # Gians
        self.kv = pf_control_params['kv']               # Vel. Gain         (sim = 0.8)
        self.kw = pf_control_params['kw']               # Angular Vel. Gain (sim = 5)
        self.k_rot = pf_control_params['k_rot']         # Heading Sensitive (sim = 5)

        # trajectory
        self.Tr = pf_control_params['Tr']
        
        # Aux. 
        self.l_width = pf_control_params['l_width']     # robot width (0.105)
        self.Ts = pf_control_params['Ts']
        self.idx = 0
        

        # Vector State
        self.v = []                                     # Vector for Tr. Segment 
        self.dn = 0                                     # # Normalized orthogonal distance
        

    def dot_mut(self, a, b):

        return a[0]*b[0] + a[1]*b[1] 

    
    def vector_state(self, state):
        '''
            Args: 
                state : Current robot position
        '''
        
        # guideline
        dx = self.Tr[self.idx+1][0] - self.Tr[self.idx][0]
        dy = self.Tr[self.idx+1][1] - self.Tr[self.idx][1] 


        # State vectors
        self.v = [dx, dy]                       # Direction vector of the current segment
        v_ort = [dy, -dx]                       # Orthogonal direction vector of the current segment
        rx = state[0] - self.Tr[self.idx][0]
        ry = state[1] - self.Tr[self.idx][1]
        r = [rx, ry]

        # Update the Trajectory Segment        
        u = self.dot_mut(self.v, r)/self.dot_mut( self.v, self.v )          # u = v.'*r/(v.'*v) from Matlab

        num_points = len(self.Tr)
        if u > 1 :
            if self.idx < num_points-2 :
                self.idx = self.idx + 1

        # Normalized orthogonal distance        
        self.dn = self.dot_mut(v_ort, r) / self.dot_mut(v_ort, v_ort)       # dn = v_ort.'*r/(v_ort.'*v_ort); 

        # T_current = [Tr(idx+1, 1); Tr(idx+1, 2)]


    def wrapTopi(self, angle):

        if angle > math.pi :
            angle = angle - 2*math.pi

        elif angle < -math.pi :
            angle = angle + 2*math.pi
        
        return angle


    def error(self, state):
        '''
            Args: 
                state : Current robot position

            Requiere: 'vector_state()'

                 v  : Segment vector
                 dn : Norm. Distance to segment
        '''

        #  Orientation of the line segment
        phi_lin = math.atan2(self.v[1], self.v[0])

        # If we are far from the line then we need
        # additional rotation to face towards the line. If we are on the left
        # side of the line we turn clockwise, otherwise counterclock wise.
        phi_rot = math.atan(self.k_rot*self.dn); 


        # Ref. Angle
        phi_ref = phi_lin + phi_rot
        phi_ref = self.wrapTopi(phi_ref)

        phi_e = phi_ref - state[2]
        phi_e = self.wrapTopi(phi_e)

        return phi_e
    

    def step(self, state, vis=False):
        
        self.vector_state(state)
        phi_e = self.error(state)

        vel = self.kv*math.cos(phi_e);             # 0.8
        w = self.kw*phi_e;                           # 5


        # Output
        vr, vl = self.output_format(vel, w)


        # Vis.
        if vis :
            print("Vels. Controller (i.e. vr, vl, v, w) = ", vr, vl, vel, w)
        
        return vr, vl
    
    
    def output_format(self, v, w):

        # Model Equations
        # V = (vr + vl)/2;
        # w = (vr - vl)/l_width;
        #
        # vr = l_width*w + vl
        # V = (l_width*w + vl + vl)/2;

        # Solution
        vl = (2*v - self.l_width*w)/2
        vr = self.l_width*w + vl

        return vr, vl