import pickle 
import datetime
import numpy as np



def save_runData_pickle(id, state, tracjectory, route_id, brake_actived, obs_detected, path="./"):
    '''
        Save date just for one Agent in a file
        with pickle data type

        (*) Take tracjectory from PSO (numpy_type)
    '''

    data = {}
    data['id'] = id
    data['state'] = state    
    data['tracjectory'] = tracjectory
    data['route_id'] = route_id
    data['brake_actived'] = brake_actived
    data['obs_detected'] = obs_detected

    # Create File
    current_time = datetime.datetime.now()
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    date_name = str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute)
    file_name = "run_data_" + date_name + "_agent_" + str(id)
    
    file_name = path + file_name 
    data_file = open(file_name, 'ab')

    # Store
    pickle.dump(data, data_file)                    
    data_file.close()

    return file_name

def load_runData_pickle(file_name):
    '''
        Return DIctionary with each Element
    '''

    data_file = open(file_name, 'rb')    
    data = pickle.load(data_file)
    
    data_file.close()

    return data


def joint_swarm_data(agents_data_list):
    '''
        (*) Joint States in a List

        Input State:
        state_1 = np.zeros((iterations, 3))
        state_2 = np.zeros((iterations, 3))

        brake_actived (iterations, 1)
        obs_detected  (iterations, 1)

    '''

    route_agent_ids = []
    state_storage_list = []
    data_list = [] 
    brake_actived_list = []
    obs_detected_list = []

    largest = 0
    smallest = 65000

    data_jointed = {}

    for i, data_dict in enumerate(agents_data_list):
        
        # IDs
        route_agent_ids.append( data_dict['id'] )

        # Joint States in a List        
        state = data_dict['state']
        state_storage_list.append( state ) 

        num_iterations = len(state)
        if num_iterations > largest :
            largest = num_iterations

        if num_iterations < smallest :
            smallest = num_iterations

        # Save numpy trajectories
        data_list.append( data_dict['tracjectory'] )

        # Collision signals
        brake_actived_list.append( data_dict['brake_actived'] )
        obs_detected_list.append( data_dict['obs_detected'] )


    data_jointed['route_agent_ids'] = route_agent_ids
    data_jointed['state_storage_list'] = state_storage_list
    data_jointed['data_list'] = data_list
    data_jointed['brake_actived_list'] = brake_actived_list
    data_jointed['obs_detected_list'] = obs_detected_list

    data_jointed['largest'] = largest
    data_jointed['smallest'] = smallest
    data_jointed['num_agents'] = i+1

    return data_jointed





# [Previous Idea]

    # Input State:
    #     state_1 = np.zeros((iterations, 1, 3))
    #     state_2 = np.zeros((iterations, 1, 3))

    #     b = np.hstack((a1, a2))
    #     b => (total_iter+1, num_agents, 3)


def save_pickle(name, dict_data, path="./"):
    '''
        name = scen_name_mothod
    '''

    # Create File
    file_name = name
    
    file_name = path + file_name 
    data_file = open(file_name, 'ab')

    # Store
    pickle.dump(dict_data, data_file)                    
    data_file.close()

    print("Pickle File Saved as ", file_name)

    return file_name

def save_time_data_pickle(name, time_dict, path="./"):
    '''
        name = scen_name_mothod
    '''

    # Create File
    file_name = name
    
    file_name = path + file_name 
    data_file = open(file_name, 'ab')

    # Store
    pickle.dump(time_dict, data_file)                    
    data_file.close()

    print("Pickle File Saved as ", file_name)

    return file_name


def load_time_data_pickle(file_name):
    '''
        Return DIctionary with each Element
    '''

    data_file = open(file_name, 'rb')    
    data = pickle.load(data_file)
    
    data_file.close()

    return data