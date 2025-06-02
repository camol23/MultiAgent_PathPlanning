#!/usr/bin/env python3

'''
    Objects to Control Locally the EV3
'''

import time
import numpy as np
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor import INPUT_2
from ev3dev2.sensor.lego import UltrasonicSensor

from control.aux_controllers import PID


class ev3_local_object:
    def __init__(self):
        
        self.id = None

        # Motor         
        self.motor_right = None
        self.motor_left = None

        # Ultrasonic
        self.ultrasonic_sensor = None

        # Ev3 Params
        self.max_vel = 200  # [deg/s]  # 20 duty cycle


        # Config.
        self.speed_p = None
        self.speed_i = None
        self.speed_d = None

        # Controller
        self.speed_pid_module_m1 = None
        self.speed_pid_module_m2 = None
        self.spped_PID_params = None      



    def initialize(self, ev3_params):
        
        self.id = ev3_params['id']        
        self.max_vel = ev3_params['max_vel']

        # Use the LargeMotor
        self.motor_right = LargeMotor(OUTPUT_C)
        self.motor_left  = LargeMotor(OUTPUT_B)

        # Ultrasonic        
        self.ultrasonic_sensor = UltrasonicSensor(INPUT_2)        
        
        # Config.
        self.speed_p = ev3_params['speed_p']
        self.speed_i = ev3_params['speed_i']        
        self.speed_d = ev3_params['speed_d']

        # Controller
        self.spped_PID_params = ev3_params['spped_PID_params']

        if self.spped_PID_params != None:
            self.speed_pid_module_m1 = PID()            
            self.speed_pid_module_m2 = PID()            
            self.speed_pid_module_m1.initialize(self.spped_PID_params)
            # self.spped_PID_params['kp'] = 20
            # self.spped_PID_params['ki'] = 0
            # self.spped_PID_params['kd'] = 0
            self.speed_pid_module_m2.initialize(self.spped_PID_params)
        


    def motors_pid_run_forever(self, vr, vl, Ts=None):
        '''
            Input : [deg/s]
        '''
        
        vr_y, vl_y = self.read_speed()
        vr = self.speed_pid_module_m1.step(vr_y, vr, Ts)
        vl = self.speed_pid_module_m2.step(vl_y, vl, Ts)
        
        vr = vr*self.max_vel
        vl = vl*self.max_vel
        self.motors_run_forever(vr, vl)


    def motors_run_forever(self, vr, vl):
        '''
            Input : [deg/s]
        '''

        vr, vl = self.limit_vels(vr, vl)

        self.motor_right.speed_sp = vr
        self.motor_left.speed_sp = vl
        self.motor_right.run_forever()
        self.motor_left.run_forever()


    def motors_stop(self):
        self.motor_right.stop()
        self.motor_left.stop()

    def read_speed(self):
        vr = self.motor_right.speed
        vl = self.motor_left.speed

        return vr, vl


    def read_ultrasonic_in_m(self):
        sensor_dist_1 = self.ultrasonic_sensor.distance_centimeters
        sensor_dist_1 = sensor_dist_1/100        

        return sensor_dist_1


    def limit_vels(self, vr, vl):
        
        vr = max( min(self.max_vel, float(vr) ), 0)
        vl = max( min(self.max_vel, float(vl) ), 0)

        return vr, vl


    def set_speed_cte_pid_internal(self):

        self.motor_right._speed_p = self.speed_p
        self.motor_right._speed_i = self.speed_i
        self.motor_right._speed_d = self.speed_d

        self.motor_left._speed_p = self.speed_p
        self.motor_left._speed_i = self.speed_i
        self.motor_left._speed_d = self.speed_d

        print("Speed PID internal ", self.motor_right._speed_p, self.motor_right._speed_i, self.motor_right._speed_d)