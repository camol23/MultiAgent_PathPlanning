#!/usr/bin/env python3

from logging_data import store_data
import numpy as np

# For DRL model in numpy
params_name = './logging_data/actor_all_18_np_params_list'
data_dict = store_data.load_time_data_pickle(params_name)
w_params_list = data_dict['w_params']
b_params_list = data_dict['b_params']

w_params = [np.array(data, dtype=np.float32) for data in w_params_list]
b_params = [np.array(data, dtype=np.float32) for data in b_params_list]
print(len(w_params))
print(b_params[0])