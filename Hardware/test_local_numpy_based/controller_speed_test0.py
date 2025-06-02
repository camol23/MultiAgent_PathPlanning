
#!/usr/bin/env python3

'''
    Motor drives for Duty Cycle
'''

import time
import numpy as np
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C


from ev3_utils.ev3_navigation import ev3_local_object


# cd test_local_numpy_based; sudo chmod 777 controller_speed_test0.py; python3 controller_speed_test0.py
# kp 0.002 - 0.003
# kp 0.00025 ki 0.008
# 0.0001    0.01    0.000_005

print("Introduce Kp")
kp = float(input())
print("Introduce Ki")
ki = float(input())
print("Introduce Kd")
kd = float(input())

print("Iterations")
iterations = int(input())



Ts = 0.4
spped_PID_params = {
    'kp' : kp,
    'ki' : ki, 
    'kd' : kd,
    
    'Ts': Ts,
    'saturation_flag' : False,
    'saturation_max' : None,
    'saturation_min' : None
}

ev3_params = {
     'id' : 0,
    'max_vel' : 300, # [deg/s]
        
    # Config.
    'speed_p' : 400,
    'speed_i' : 1200,
    'speed_d' : 5,

    # Controller
    'spped_PID_params' : spped_PID_params
}


ev3_object = ev3_local_object()
ev3_object.initialize(ev3_params)


vr_list = []
vl_list = []
u_list = []

run_time_list = []

ref = 200
run_time = 0
print("running with ref = ", ref)
# iterations = 15
for i in range(0, iterations):

    start_time = time.time()
    ev3_object.motors_pid_run_forever(ref, ref, Ts+run_time)

    vr, vl = ev3_object.read_speed()
    # print("Speed = ", vr, vl)    
    run_time_list.append(run_time)
    vr_list.append(vr)
    vl_list.append(vl)

    run_time = time.time() - start_time     
    time.sleep(Ts)


# Measures
vr_np = np.array(vr_list)
vl_np = np.array(vl_list)

vr_mean = np.mean(vr_np)
vl_mean = np.mean(vl_np)
vr_std = np.std(vr_np)
vl_std = np.std(vl_np)
vr_max = np.max(vr_np)
vl_max = np.max(vl_np)
vr_min = np.min(vr_np)
vl_min = np.min(vl_np)

print("running with ref = ", ref, iterations, Ts)
print("run time Av. = ", run_time)
print("Vr Av. STD max min ", vr_mean, vr_std, vr_max, vr_min)
print("Vl Av. STD max min ", vl_mean, vl_std, vl_max, vl_min)

print("Stop")
ev3_object.motors_stop()