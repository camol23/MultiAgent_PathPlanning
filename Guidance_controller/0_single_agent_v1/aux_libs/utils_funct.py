import os
import numpy as np



def vis_training_config(data_log):
    ''''
    
    '''
    print("--- Training Configuration --- ")
    print()
    for data_log_dict in data_log:
        for key, value in data_log_dict.items():            
            print(key, value)

        print()
        

