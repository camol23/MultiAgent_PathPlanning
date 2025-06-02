import math
import numpy as np
from utils_fnc.op_funct import distance, angle_tan2



def limit_speed(agent_state, states, agent_idx, margin=10):
    
    distances = []
    angles = []
    for i in range(0, states.shape[0]):

        if i != agent_idx :
            distance_i = distance(states[i, 0], states[i, 1], agent_state[0], agent_state[1])                    
            angle_i = angle_tan2(agent_state[0], agent_state[1], states[i, 0], states[i, 1])
            
            distances.append( distance_i )
            angles.append( abs(angle_i) )

            # print(i, distance_i, states[i, 0], states[i, 1], agent_state[0], agent_state[1])

    # Trun to Numpy
    angle_array = np.array(angles)
    dist_array = np.array(distances)

    # Discarted by Angle
    angle_range = 80
    dist_chosen = np.where(angle_array<math.radians(angle_range), dist_array, dist_array*2*margin )

    min_dist = np.min(dist_chosen)    
    min_dist = min_dist/margin

    if min_dist <= 0.2:
        val = 0    
    else:
        val = -math.exp(-5*abs(min_dist))+1
    
    # Is close 
    if val <= 0.99 :
        active_flag = 1
    else:
        active_flag = 0
    

    # print("Brake factor = ", val, min_dist*margin)
    # print()
    return val, active_flag

