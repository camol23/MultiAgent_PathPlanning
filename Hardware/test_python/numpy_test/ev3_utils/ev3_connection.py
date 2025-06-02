#!/usr/bin/env python3

import rpyc


class ev3_object:
    def __init__(self):
        
        self.conn = None
        self.IP = None

        # Motor 
        self.ev3dev2_motor_C = None
        self.ev3dev2_motor_B = None
        self.motor_right = None
        self.motor_left = None


        # Ultrasonic
        self.ev3dev2_sensor = None
        self.ev3dev2_sensor_lego = None
        self.ultrasonic_sensor = None


        # Ev3 Params
        self.max_vel = 200


    def initialize(self, ev3_params):
        
        self.IP = ev3_params['IP']
        self.conn = rpyc.classic.connect(self.IP)                # ev3dev

        # import ev3dev2 on the remote ev3dev device
        self.ev3dev2_motor_C = self.conn.modules['ev3dev2.motor']
        self.ev3dev2_motor_B = self.conn.modules['ev3dev2.motor']

        # Use the LargeMotor and TouchSensor on the remote ev3dev device
        self.motor_right = self.ev3dev2_motor_C.LargeMotor(self.ev3dev2_motor_C.OUTPUT_C)
        self.motor_left  = self.ev3dev2_motor_B.LargeMotor(self.ev3dev2_motor_B.OUTPUT_B)        

        # Ultrasonic
        self.ev3dev2_sensor = self.conn.modules['ev3dev2.sensor']
        self.ev3dev2_sensor_lego = self.conn.modules['ev3dev2.sensor.lego']
        self.ultrasonic_sensor = self.ev3dev2_sensor_lego.UltrasonicSensor(self.ev3dev2_sensor.INPUT_2)

        # Params
        self.max_vel = ev3_params['max_vel']
        


    def motors_run_forever(self, vr, vl):
        '''
            Input : [deg/s]
        '''

        # vr, vl = self.limit_vels(vr, vl)
        self.motor_right.run_forever(speed_sp=vr)
        self.motor_left.run_forever(speed_sp=vl)


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
