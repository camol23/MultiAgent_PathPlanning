#!/usr/bin/env python3


import numpy as np
import time


print("Program Start")
init_t = time.time()
b = np.array([1, 2, 3, 4])
c = np.array([10, 20, 30, 40])
init_t = time.time() - init_t

op_time = time.time()
d = np.where(b<3, c, c*100)
op_time = time.time() - op_time
print(d)
print("Time init ", init_t)
print("Time Op ", op_time)
