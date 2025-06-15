
#!/usr/bin/env python3

'''
    cat /sys/class/tacho-motor/motor0/speed_pid/Kp
    kp 1000
    ki 60
    kd 0

'''


import time
import numpy as np
from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C
# from ev3dev2.motor import Motor, OUTPUT_B, OUTPUT_C

from utils_fnc import op_funct
from logging_data import store_data


# --------------
num_iter = 100
Ts = 0.01
# --------------

motor_right = LargeMotor(OUTPUT_C)
motor_left = LargeMotor(OUTPUT_B)
# motor_right = Motor(OUTPUT_C)
# motor_left = Motor(OUTPUT_B)
motor_right.reset()
motor_left.reset()
motor_right.stop()
motor_left.stop()

# print("regulation val = ", motor_right.speed_regulation_enabled)
# print("speed_regulation_p = ", motor_right.speed_regulation_p)
print()



# motor_right.speed_sp = SpeedDPS(200)
# rotates the motor at 200 RPM (rotations-per-minute) for five seconds.
# LargeMotor.on_for_seconds(SpeedDPS(200), 5)


#400, 1200, 5, 23, 5, 0
print("Speed PID", motor_right._speed_p, motor_right._speed_i, motor_right._speed_d)
# motor_right._speed_p = 400
# motor_right._speed_i = 1200
# motor_right._speed_d = 5

# motor_left._speed_p = 400
# motor_left._speed_i = 1200
# motor_left._speed_d = 5


# motor_right.speed_p = 400 
# motor_right.speed_i = 1200
# motor_right.speed_d = 5

# motor_left.speed_p = 400
# motor_left.speed_i = 1200
# motor_left.speed_d = 5
print("Speed PID", motor_right._speed_p, motor_right._speed_i, motor_right._speed_d)

vel_list = []
vel_left_list = []

print("running ...")
motor_right.speed_sp = 200
motor_left.speed_sp = 200
motor_right.run_forever()
motor_left.run_forever()
for i in range(0, num_iter):

    vel_list.append(motor_right.speed)
    vel_left_list.append(motor_left.speed)
    # print("Speed = ", motor_right.speed)
    time.sleep(Ts)


print("Speed R PID", motor_right._speed_p, motor_right._speed_i, motor_right._speed_d)
print("Speed L PID", motor_left._speed_p, motor_left._speed_i, motor_left._speed_d)
print("Stop")
motor_right.stop()
motor_left.stop()

vel_np = np.array(vel_list)
vel_mean = np.mean(vel_np)
vel_std = np.std(vel_np)
vel_var = np.var(vel_np)
print("Vel len", len(vel_list))
print("Vel range ", np.min(vel_np), np.max(vel_np))
print("Vel mean ", vel_mean)
print("Vel STD ", vel_std)
print("Vel Var. ", vel_var)
print()
print("Left Motor")
print("Vel range ", np.min(vel_left_list), np.max(vel_left_list))
print("Vel mean ", np.mean(vel_left_list))
print("Vel STD ", np.std(vel_left_list))
print("Vel Var. ", np.var(vel_left_list))
print()
print("Joint Data")
print("Vel range ", np.min(vel_left_list+vel_list), np.max(vel_left_list+vel_list))
print("Vel mean ", np.mean(vel_left_list+vel_list))
print("Vel STD ", np.std(vel_left_list+vel_list))
print("Vel Var. ", np.var(vel_left_list+vel_list))
print()

r_width = 0.056/2
vel_r_m_list = [op_funct.deg2m(deg_val, r_width) for deg_val in vel_list]
print("R m/s ")
print("Vel mean ", np.mean(vel_r_m_list))
print("Vel STD ", np.std(vel_r_m_list))
print("Vel Var. ", np.var(vel_r_m_list))
print()
vel_l_m_list = [op_funct.deg2m(deg_val, r_width) for deg_val in vel_left_list]
print("L m/s ")
print("Vel mean ", np.mean(vel_l_m_list))
print("Vel STD ", np.std(vel_l_m_list))
print("Vel Var. ", np.var(vel_l_m_list))


# Prepaer Data to be saved
dict_data = {
    'vel_list' : vel_list,
    'vel_left_list' : vel_left_list,
}

# path = "./logging_data/time_data_exp/"
path = '/home/robot/test_local_numpy_based/logging_data/measures/'
name = "motor_R_L_200_not_usable"
store_data.save_pickle(name, dict_data, path)