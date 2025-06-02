import sys
import os
import copy
import time
import numpy as np

from Env import env_v1
from DRL import PPO_v2
from aux_libs import store_model
from aux_libs import utils_funct
from aux_libs import scheduler_aux



# --------------------------------- Configuration Section ---------------------------------------------------
# ___________________________________________________________________________________________________________

plot_title = "Test 2 - heading exp. #2 + dist as states + time -0.0001 + coins 0.01*6p + update50 + guideline state"
# + coins 0.01*6p
exe_type = {
    # Execution Type 
    'training_phase' : True,           # False: Load the previous trained model
    'model_type'    : 'PPO',           # Choose between PPO or A3C (Not avaible yet)
    
    # Steps config.
    'num_episodes' : 3000,              # Training Episodes
    'max_steps' : 600,                   # Max. STEPS by Episode (rollout)

    # Update Training
    'update_during_steps' : True,       # Update grads during rollout
    'update_step' : 200,                # Steps to Update model 

    # Steps schecduler
    'scheduler_steps' : False,           # The steps are constrained and evolve over the time
    'init_val' : 300,                   # Min. STEPS for scheduler 
    'div_episodes' : 2,                 # Divide num_episodes to do the sequence

    # env
    'normilize_states_env' : True       # Normalize state functions [0 - 1]
}


# Training Parameters and Configuration Settings
train_config = {
        'gamma' : 0.99,
        'epsilon' : 0.2,                            # PPO clipping parameter
        'epochs' : 10,                              # Number of epochs per update
        'batch_size' : 64,                          # Mini-Batch size
        'norm_returns' : True,                      # Normalize computed returns
        'entropy_coef' : 0.005,                      # Entropy Coeffitient for Pi_loss
        'clip_grad_val' : 10,                      # Clipping value for Gradients
        'return_type' : 'gae',                   # Choose between (classic return, n-step boostrap, TDT, GAE)
        'val_loss_coef' : 1,                        # Factor to multiply val. loss
        'gae_lambda': 0.95
}

num_epi = exe_type['num_episodes']

# Define model architecture
model_architecture = {
        # Model  in/out
        'model_id' : 1,                             # Choose between different predefined architectures on networks_ppo_v1
        'state_dim' : 3,                            # Num. States
        'action_dim' : 3,                           # Num. Actions
        'hidden_dim' : 128,                          # Num. Hidden Layers
        'active_states' : ['dist',
                           'dist_guideline', 
                           'headinge'],             # state[0, 1]
                        #    'orientation' ],         # Select which states are goin to be read  ['dits', 'dist_guideline', 'orientation']
        
        'opti_type'  : 'adamw',                     # Choose between Optimizer Type (adamw, adam)
        
        # Learning rate
        'lr_rate' : 1e-3,                           # Lr. Base for Actor and Critic
        'critic_coef_lr' : 0.5,                     # lr*critic_coef (It's recommended has a smaller lr_critic)                      
        'lr_critic' : 0,                            # Lr. Just for Critic. if = 0 it's applied the critic_coef
        'lr_scheduler_type' : 'cosine',             # Activate Cosine Scheduler with 'cosine'

        # Cosine Scheduler         
        'warmup_epochs'  : int(num_epi*0.05),        # Ramp time 
        'num_episodes'   : num_epi
}


max_steps_exa = exe_type['max_steps']

# Trining parms. Env.
map_training_params = {
    'goal_pos' : (220, 480),                        # Goal coordinates
    'goal_tolerance' : 0.04,                        # Goal Margin Zone (around the goal point)
    'max_steps' : max_steps_exa,                    # Max. Steps by Episode (rollout)

    'section_type' : '',                            # (div_segments) The goal is moving towards the guideline (env.load())
    'wait_in_wp'     : 20,                           # Number of times to reach goal to pass to the next one
    'path_points'    : 3                            # Num. of WP in the Defined path 
}



##### Common Sim. Settings

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

# The Dictionaries in the list are log in a .txt file in the end of the test
log_data_list = [exe_type, train_config, model_architecture, map_training_params, agents_settings, map_settings]
# --------------------------------------------------------------------------------------------------------



# Initialize Environment
training_phase = exe_type['training_phase']
env = env_v1.Environment(map_settings, agents_settings, training_flag=training_phase)
env.initialize(map_training_params)

# Load the Path (2-Points)
path = np.transpose(np.array([agents_settings['start_pos'], map_training_params['goal_pos'] ]))
env.load_path(path, map_training_params['section_type'], map_training_params['path_points'])
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# Create the Model
model = PPO_v2.PPO_model()                      # Object witout network
model.initialize(model_architecture)            # Load Networks and Optimizer


# Load a pretrained Model
training_phase = exe_type['training_phase']
if not training_phase :
    checkpoint_dict = store_model.load_model_forTest(model, model.splitNets_flag, folder_number=None, checkpoint_name='')



# Time measurements
time_spend_step = 0
time_spend_train = 0
start_time_sim = time.time()

# Training Loop
normilize_states_env = exe_type['normilize_states_env']
if exe_type['scheduler_steps'] :
    steps_val = scheduler_aux.scheduler_linear(exe_type)
    steps_val.init()
    env.max_steps = steps_val.init_val


for i in range(0, num_epi):
    
    done = True
    env.reset_env()
    
    # clear Data and take current state 
    model.mem_trajectory.reset(env)

    
    env.global_iterations = i
    start_time = time.time()
    counter_steps = 0

    while done :
        
        # Policy 
        actions, log_probs = model.get_action(model.mem_trajectory.states, training_phase)
        
        # Env. Step
        env.apply_one_action_left_right(actions)
        env.env_step(normalize_states=normilize_states_env)
        done = not(env.stop_steps)
                
        # Store Trajectory (Batch)
        model.mem_trajectory.store_trajectory(env, actions, log_probs, not(done))

        # env.visuzalization()
        if not(env.running_flag):
            print("Stopped by user")
            break
        
        counter_steps += 1
        if training_phase :
            update_flag = (counter_steps % exe_type['update_step']) == 0

            if exe_type['update_during_steps'] and update_flag :
                print("TRAIN in Steps ---", env.steps, done)
                if model.splitNets_flag :
                    model.train_splitNets(train_config)
                else:    
                    model.train(train_config)

            
    
    print("reward_steps", env.reward_steps)
    # Time measure
    end_time = time.time()
    time_spend_step = end_time - start_time

    # MODEL UPDATE                 
    if training_phase :
        if model.splitNets_flag :
            model.train_splitNets(train_config)
        else:    
            model.train(train_config)

    # Steps scheduler update
    if exe_type['scheduler_steps'] :
        steps_val.update()
        env.max_steps = steps_val.val

    # Time meausure
    time_spend_train = time.time() - end_time
    if i == 0 :
        time_spend_step_mean = time_spend_step
        time_spend_train_mean = time_spend_train 
    time_spend_step_mean = (time_spend_step_mean + time_spend_step)/2.0
    time_spend_train_mean = (time_spend_train_mean + time_spend_train)/2.0

    # Vis.
    print("STEPS = ", env.steps)
    print("Execution Time: Rollout = {} - Training {} - Total = {} ".format(time_spend_step, time_spend_train, time_spend_step + time_spend_train ) )
    print("______________________________________________")
    if not(env.running_flag):
        print("Stopped by user")
        break
    num_episodes_done = i

# END OF TRAINING
training_time_minutes = (time.time() - start_time_sim)/60.0 

# Plot Training Results
if training_phase :
    model.plot_training(episodes=num_episodes_done, steps=env.max_steps, title=plot_title)



# Vis. Training Data
print()
print("Ave. Execution time cycle: Rollout = {} [s] - Training = {} [s] ".format(time_spend_step_mean, time_spend_train_mean) )
print("Total Training time = {} [min] ".format( training_time_minutes ) )

print()
print("Plot Title = ", plot_title)
print("Checkpoint Path", model.folder_path)
if training_phase :
    utils_funct.vis_training_config(log_data_list)

# Saved the last model
if training_phase :
    store_flag = input("Do you wanna Store the Model? y/n ... ")
else:
    store_flag = 'n'

if (store_flag == 'y') :
    
    # include data in log.txt file
    add_data = {
        'training_time_minutes' : training_time_minutes,
        'time_spend_step' : time_spend_step,
        'time_spend_train' : time_spend_train,
        'checkpoint_path' : model.folder_path,
        "splitnet" : model.splitNets_flag
    }
    log_data_list.append( add_data )
    store_model.save_modelFromTraining(model, model.splitNets_flag, model.mem_trajectory.rewards_steps, log_data_list)






# END
sys.exit()