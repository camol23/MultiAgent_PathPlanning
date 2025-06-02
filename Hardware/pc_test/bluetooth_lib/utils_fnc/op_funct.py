#!/usr/bin/env python3

import math



# Sample time
# Should be a Class

# Distance functions
def distance(xa, ya, xb, yb):
    sum = (xa - xb)**2 + (ya - yb)**2
    
    return math.sqrt(sum)



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