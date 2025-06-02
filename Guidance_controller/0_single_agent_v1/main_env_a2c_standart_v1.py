import sys
import os
import numpy as np

from Env import env_v1
from DRL import a2c_standart_v1
from aux_libs import store_model


# ----- Execution Type -----

testing_exe = False     # Load a Model and disable Traning 
#training_exe = True
#store_flag = False

# --------------------------




# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (100, 550),                    #(50, 550),
    'num_agents': 1,
    'formation_type': 2                         # 2: V formation
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    'num_obs': 0,
    'type_obs': 'random',                       # Random sqaure obstacles
    'seed_val_obs': 286,                        # Test obstacles location
    'mouse_flag': True,                         # Mouse pointer is turned in a sqaere obstacle
    'max_rect_obs_size': 200                    # maximun Obstacle size
}


# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings, training_flag=True)
env.initialize()

# goal_pos = (700, 300) #(1000, 200)
goal_pos = (220, 480)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# DRL model
state_dim = 3       # [Dist. to goal, Dist. to guidline]
action_dim = 2      # [rigth, left, nothing]
model = a2c_standart_v1.ActorCritic_Agent(state_dim=state_dim, action_dim=action_dim)

# Training Parameters
num_iterations = 200
env.max_steps = 200

# When apply backpro
# num_iter_to_train = int(env.max_steps/2)
num_iter_to_train = 2


for i in range(0, num_iterations):

    # *It shoud be in a reset function*
    states = np.zeros((1, state_dim))
    states_steps = np.zeros((1, state_dim))
    next_state = np.zeros((1, state_dim))
    next_state_steps = np.zeros((1, state_dim))
    rewards = np.zeros((1, 1))
    rewards_steps = np.zeros((1, 1))
    # actions = np.zeros((1, 1))
    actions_steps = np.zeros((1, 1))

    env.reset_env()
    env.get_output_step(normalize_states = True)
    states[0, 0] = env.state_distance[-1][-1]
    states[0, 1] = env.state_dist_to_guideline[-1][-1]
    states[0, 2] = env.state_orientation[-1][-1]

    done = False
    env.global_iterations = i
    counter_to_train = 0

    while not done :
        
        actions, _ = model.select_action(states, vis_flag=True, vis_vals=True)
        env.apply_one_action_left_right(actions)
        env.env_step(normalize_states=True)

        next_state[0, 0] = env.state_distance[-1][-1]
        next_state[0, 1] = env.state_dist_to_guideline[-1][-1]
        next_state[0, 2] = env.state_orientation[-1][-1]

        rewards[0, 0] = env.reward_total_list[-1]
        # rewards[0, 0] = rewards[0, 0] + env.reward_total_list[-1]
        
        done = env.stop_steps
        env.visuzalization()

        # Store batch
        states_steps = np.vstack( (states_steps, states) )
        next_state_steps = np.vstack( (next_state_steps, next_state) )
        rewards_steps = np.vstack( (rewards_steps, rewards) )
        actions_steps = np.vstack( (actions_steps, actions) )

        # Stop by Keyboard command "Q"
        if not(env.running_flag):
            print("Stopped by user")
            break

        if (counter_to_train >= num_iter_to_train) or (done) :
            
            # Remove the first row (init. array with zeros)
            states_steps = np.delete( states_steps, 0, axis=0)
            next_state_steps = np.delete( next_state_steps, 0, axis=0)
            rewards_steps = np.delete( rewards_steps, 0, axis=0)
            actions_steps = np.delete( actions_steps, 0, axis=0)

            # model.train_step(states, actions, rewards, next_state, done, vis_flag=False)
            model.train_batch(states_steps, actions_steps, rewards_steps, next_state_steps, done, vis_flag=False)
            counter_to_train = 0

            # Reset Batch
            # *It shoud be in a reset function*
            states_steps = np.zeros((1, state_dim))
            next_state_steps = np.zeros((1, state_dim))
            rewards_steps = np.zeros((1, 1))            
            actions_steps = np.zeros((1, 1))
        else:
            counter_to_train = counter_to_train + 1             

        # Update
        states = next_state
    


    # Stop by Keyboard command "Q"
    if not(env.running_flag):
        print("Stopped by user")
        break
        


# model.plot_rewards()
model.plot_training(episodes=num_iterations)


print()
if (not testing_exe) :
    store_flag = input("Do you wanna Store the Model? y/n ... ")

if (store_flag == 'y') :

    # Count all files in a directory
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_standart'    
    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)
    num_name = num_files + 1 
    
    store_model.save_model(model.network, model.optimizer, num_name, rewards_steps, folder_path)



sys.exit()