#!/usr/bin/env python3

'''
    Common Controller Modules
'''


class PID:
    def __init__(self):
                
        self.kp = None
        self.ki = None
        self.kd = None

        self.error = None
        self.error_nm1 = None
        self.proportional = None
        self.integral = None
        self.derivative = None
        
        self.ref = 0
        self.output = 0

        # Auxliar
        self.Ts = None
        self.saturation_flag = None
        self.saturation_max = None
        self.saturation_min = None


    def initialize(self, pid_params):

        self.kp = pid_params['kp']
        self.ki = pid_params['ki']
        self.kd = pid_params['kd']

        self.Ts = pid_params['Ts']
        self.saturation_flag = pid_params['saturation_flag']
        self.saturation_max = pid_params['saturation_max']
        self.saturation_min = pid_params['saturation_min']

        self.error = 0
        self.error_nm1 = 0
        self.proportional = 0
        self.integral = 0
        self.derivative = 0
        

    def step(self, y_input, ref, Ts=None):

        if Ts == None:
            Ts = self.Ts # Fix Ts        

        # PID Computation
        self.error = ref - y_input
        self.proportional = self.error
        self.integral = self.integral + Ts*self.error
        self.derivative = (self.error - self.error_nm1)/Ts

        u = self.kp*self.proportional + self.ki*self.integral + self.kd*self.derivative

        if self.saturation_flag :
            u = self.saturation(u)
        
        return u


    def saturation(self, val):

        val_sat = max( min(self.saturation_max, val ), self.saturation_min)
        return val_sat

    


