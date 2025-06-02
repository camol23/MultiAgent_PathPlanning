import sys
import os
import copy
import time
import numpy as np


from Env import env_v1
from DRL import PPO_v1
from aux_libs import store_model
from DRL.networks import networks_a2c_v1_2

from aux_libs import scheduler_aux
from aux_libs import learning_scheduler


# from DRL.networks import networks_ppo_v1
# print(networks_ppo_v1.MODELS_PPO_LIST)
# --------------------------------------- Execution Type ---------------------------------------------------
title_test = "Test "

testing_exe = False             # Load a Model and disable Traning 
                                # If FALSE the model is in Training phase
splitNets_flag = True           # It's trained a network for the actor and another for the critic

# Training Parameters
num_iterations = 200             # Num. Episodes
max_steps = 600                  # Max. Steps by Episode
min_steps = 10                   # Min.
lapse_steps = 200                # Episodes before to increase the num.


# Training Parameters and Configuration Settings
train_config = {
        'gamma' : 0.99,
        'epsilon' : 0.2,                            # PPO clipping parameter
        'epochs' : 10,                              # Number of epochs per update
        'batch_size' : 64,                          # Mini-Batch size
        'norm_returns' : True,                      # Normalize computed returns
        'entropy_coef' : 0.01,                      # Entropy Coeffitient for Pi_loss
        'clip_grad_val' : 0.5,                      # Clipping value for Gradients
}


# Trining parms. Env.
map_training_params = {
    'goal_tolerance' : 0.03,                        # Goal zone Margin (around the goal point)
    'wait_in_wp'     : 1,                           # Number of times to reach goal to pass to the next one
    'section_type' : 'div_segments',              # Slect the WP generation with the Defined path (env.load())
    'path_points'    : 3                            # Num. of WP in the Defined path 
}

# Model
state_dim = 3
action_dim = 3
hidden_dim = 128

# Cosine Scheduler 
lr_sheduler_flag = True
warmup_epochs = int(num_iterations*0.05)
lr_rate = 1e-2

# --------------------------------------------------------------------------------------------------------


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
env = env_v1.Environment(map_settings, agents_settings, training_flag=not(testing_exe))
env.initialize()
#   Env. Training
env.max_steps = max_steps
env.wait_in_wp = map_training_params['wait_in_wp']
env.goal_tolerance = map_training_params['goal_tolerance']


# goal_pos = (700, 300) #(1000, 200)
goal_pos = (220, 480)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path, map_training_params['section_type'], map_training_params['path_points'])

print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# DRL model
model = PPO_v1.PPO_model(state_dim, action_dim, lr=lr_rate, hidden_dim=hidden_dim, splitNets_flag=splitNets_flag)

# load model
if testing_exe :

    # Count all files in a directory (Saved it when the training ends)
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_main'    
    if splitNets_flag :
        folder_path = folder_path + '/split'
    else:
        folder_path = folder_path + '/oneNet'

    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)

    if splitNets_flag :
        folder_path = folder_path + '/model_' + str(num_files)
        actor_path = folder_path + '/actor_v1/checkpoint_episode_' + str(num_files) + '.pt'
        critic_path = folder_path +  '/critic_v1/checkpoint_episode_' + str(num_files) + '.pt'
    else:
        actorCritic_path = folder_path + '/actorCritic_v1/checkpoint_episode_' + str(num_files) + '.pt'

    # IF I WANT A CHECKPOINT: 
    #       1) Uncomment the actor/critic path
    #       2) Update the Checkpoint to be tested
    #       3) Update the folder number
    folder_number = 72
    checkpoint_name = "/checkpoint_episode_174_reward_[2155.62004259].pt"

    # checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1'
    checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1_splitNets'    
    folder_name = checkpoint_path + '/model_ppo_v1_test_' + str(folder_number)
    
    # actorCritic_path = folder_name + '/actorCritic_v1' + checkpoint_name
    # actor_path = folder_name + '/actor_ppo_v1' + checkpoint_name
    # critic_path = folder_name + '/critic_ppo_v1' + checkpoint_name

    print()
    print("Model Loaded = ", actor_path)
    print()

    if splitNets_flag :
        model.actor_model, model.opt_actor, _ = store_model.load_model(model.actor_model, model.opt_actor, actor_path)
        model.critic_model, model.opt_critic, checkpoint_dict = store_model.load_model(model.critic_model, model.opt_critic, critic_path)
    else:
        model.actor_critic, model.optimizer, _ = store_model.load_model(model.actor_critic, model.optimizer, actorCritic_path)



# steps_val = scheduler_aux.scheduler_linear(min_steps, max_steps, lapse_steps, num_iterations)
# steps_val.init()


if lr_sheduler_flag and splitNets_flag :
        model.sheduler_flag = lr_sheduler_flag
        model.scheduler_actor = learning_scheduler.CosineWarmupScheduler(model.opt_actor, warmup_epochs, num_iterations)
        model.scheduler_critic = learning_scheduler.CosineWarmupScheduler(model.opt_critic, warmup_epochs, num_iterations)

time_spend_step = np.zeros((1,))
time_spend_train = np.zeros((1,))
start_time_sim = time.time()

for i in range(0, num_iterations):

    # *It shoud be in a reset function*
    states = np.zeros((1, state_dim))
    states_steps = np.zeros((1, state_dim))
    rewards = np.zeros((1, 1))
    rewards_steps = np.zeros((1, 1))
    actions_steps = np.zeros((1, 1))
    log_probs_steps = np.zeros((1, 1))

    done = True
    env.reset_env()
    env.global_iterations = i

    start_time = time.time()
    while done :
        
        actions, log_probs = model.get_action(states)
        env.apply_one_action_left_right(actions)
        env.env_step(normalize_states=True)
        
        # Save current step
        states[0, 0] = env.state_distance[-1][-1]
        states[0, 1] = env.state_dist_to_guideline[-1][-1]
        states[0, 2] = env.state_orientation[-1][-1]

        rewards[0, 0] = env.reward_total_list[-1]
        done = not(env.stop_steps)

        # Store Trajectory (Batch)
        states_steps = np.vstack( (states_steps, states) )
        rewards_steps = np.vstack( (rewards_steps, rewards) )
        actions_steps = np.vstack( (actions_steps, actions) )
        log_probs_steps = np.vstack( (log_probs_steps, log_probs) )

        # env.visuzalization()
        if not(env.running_flag):
            print("Stopped by user")
            break

    
    # Remove the first row (init. array with zeros)
    states_steps = np.delete( states_steps, 0, axis=0)
    rewards_steps = np.delete( rewards_steps, 0, axis=0)
    actions_steps = np.delete( actions_steps, 0, axis=0)
    log_probs_steps = np.delete( log_probs_steps, 0, axis=0)
    # print()
    # print(states_steps)
    # print(rewards_steps)
    # print(actions_steps)

    end_time = time.time()
    time_spend_step = np.append( time_spend_step, end_time - start_time )
                           
    if not testing_exe :
        if splitNets_flag :
            model.train_splitNets(states_steps, actions_steps, rewards_steps, done, log_probs_steps, train_config)
        else:    
            model.train(states_steps, actions_steps, rewards_steps, done, log_probs_steps, train_config)

    time_spend_train = np.append(time_spend_train, time.time() - end_time )

    print("STEPS = ", env.steps)

    time_spend_step_item = time_spend_step[-1]
    time_spend_train_item = time_spend_train[-1]
    print("Execution Time: Rollout = {} - Training {} - Total = {} ".format(time_spend_step_item, time_spend_train_item, time_spend_step_item+time_spend_train_item ) )
    # steps_val.update()
    # env.max_steps = steps_val.val
    # print("COUNTER = ", env.agents_obj[0].wait_counter)    

    del states
    del states_steps
    del rewards    
    del actions_steps 
    del log_probs_steps

    print("______________________________________________")
    if not(env.running_flag):
        print("Stopped by user")
        break


# Execution Ends
print("Ave. Execution time cycle: Rollout = {} [s] - Training = {} [s] ".format(np.mean(time_spend_step), np.mean(time_spend_train) ))
print("Total Training time = {} [min] ".format( (time.time() - start_time_sim)/60.0 ) )

# model.plot_rewards()
if not(testing_exe) :
    model.plot_training(episodes=num_iterations, steps=env.max_steps, title=title_test)



############################## Not Updated Yet ###############################
# Saved the last model
print()
print("Test about = ", title_test)
if (not testing_exe) :
    store_flag = input("Do you wanna Store the Model? y/n ... ")
else:
    store_flag = 'n'

if (store_flag == 'y') :

    # Count all files in a directory
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_main'   

    if splitNets_flag :
        folder_path = folder_path + '/split'
    else:
        folder_path = folder_path + '/oneNet'

    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)
    num_name = num_files + 1 

    # folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints'
    if splitNets_flag :
        actor_path = folder_path + '/model_' + str(num_name) + '/actor_v1'
        critic_path = folder_path + '/model_' + str(num_name) +  '/critic_v1'

        print("Saved as = ", actor_path + "_" + str(num_name) )
    else:
        actorCritic_path = folder_path + '/actorCritic_v1'

        print("Saved as = ", actorCritic_path + "_" + str(num_name) )
    

    if splitNets_flag :
        store_model.save_model(model.actor_model, model.opt_actor, num_name, rewards_steps, actor_path)
        store_model.save_model(model.critic_model, model.opt_critic, num_name, rewards_steps, critic_path)
    else:
        store_model.save_model(model.actor_critic, model.optimizer, num_name, rewards_steps, actorCritic_path)
    


sys.exit()