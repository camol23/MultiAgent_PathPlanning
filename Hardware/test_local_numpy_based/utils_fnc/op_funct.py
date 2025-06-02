#!/usr/bin/env python3

import math



# Sample time
# Should be a Class

# Distance functions
def distance(xa, ya, xb, yb):
    sum = (xa - xb)**2 + (ya - yb)**2
    
    return math.sqrt(sum)

# Angle
def angle_tan2(xa, ya, xb, yb):

    co = yb - ya
    ca = xb - xa

    return math.atan2(co, ca)


# Units Converters
def degToRad(deg_val):
    return (deg_val*math.pi)/180


def rad2m(rad_val, r):
    return rad_val*r


def deg2m(deg_val, r):
    rad_val = degToRad(deg_val)
    
    return rad2m(rad_val, r)


def deg2m_two(deg_val1, deg_val2, r):
    val1 = deg2m(deg_val1, r)
    val2 = deg2m(deg_val2, r)

    return val1, val2

def m2deg_two(m_val1, m_val2, r):

    # m to rad
    rad1 = m_val1/r
    rad2 = m_val2/r

    # rad to deg
    deg1 = (rad1*180)/math.pi
    deg2 = (rad2*180)/math.pi

    return deg1, deg2


def radToDeg(rad_val):
    return (rad_val*180)/math.pi


def is_smaller(val, margin):
    '''
        if val <= margin 
    '''
    if val <= margin :
        flag = 1
    else:
        flag = 0

    return flag


def convert_obs_coor(obs_list):
    '''
        input  = (x_botton, y_botton, rect_w, rect_h)
        output = (x_botton_left, y_botton_left, x_rigth_up, y_rigth_up)
    '''
    
    obs_list_output = []
    for i in range(0, len(obs_list)):
        rect_w = obs_list[i][0] + obs_list[i][2]    # x
        rect_h = obs_list[i][1] + obs_list[i][3]    # y
        
        obs_list_output.append((obs_list[i][0], obs_list[i][1], rect_w, rect_h)) 
        
    return obs_list_output