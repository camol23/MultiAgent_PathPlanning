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


def circle_line_intersection(l_points, circle_coor, circle_r):

    point_a = l_points[0]
    point_b = l_points[1]

    # Line 
    m = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
    b = point_b[1] - m*point_b[0]

    # Circle 
    h = circle_coor[0]
    k = circle_coor[1]
    
    # Equation params.
    a_equ = 1 + m**2
    b_equ = -2*h + 2*m*b - 2*k*m 
    c_equ = h**2 + b**2 - 2*k*b + k**2 - circle_r**2 

    # Result
    delta = b_equ**2 - 4*a_equ*c_equ

    if delta >= 0 :
        x1 = (-b_equ + math.sqrt(delta))/(2*a_equ)
        y1 = m*x1 + b

        x2 = (-b_equ - math.sqrt(delta))/(2*a_equ)
        y2 = m*x2 + b
    else:
        x1 = 0 
        y1 = 0
        x2 = 0 
        y2 = 0

    print(m)
    print(b)

    point1 = [x1, y1]
    point2 = [x2, y2]

    return  point1, point2, delta



def trans_coor(point, translation, scale):
    x = point[0]
    y = point[1]

    xt = translation[0]
    yt = translation[1]

    x_out = (x - xt)/scale
    y_out = (y - yt)/scale

    return x_out, y_out
