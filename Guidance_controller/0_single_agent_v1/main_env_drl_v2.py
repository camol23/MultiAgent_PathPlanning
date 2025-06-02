import sys
import os
import numpy as np

from Env import env_v1
from DRL import a2c_test_v1_2
from aux_libs import store_model
from DRL.networks import networks_a2c_v1_2



# ----- Execution Type -----

testing_exe = True              # Load a Model and disable Traning 
                                 # If FALSE the model is in Training phase

# Training Parameters
num_iterations = 200             # Num. Episodes
max_steps = 200                  # Max. Steps by Episode

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
env = env_v1.Environment(map_settings, agents_settings, training_flag=not(testing_exe))
env.initialize()
env.max_steps = max_steps

# goal_pos = (700, 300) #(1000, 200)
goal_pos = (220, 480)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# DRL model
model = a2c_test_v1_2.drl_model()

# Load a new Model 
# (It should be in a init. Function )
state_dim = 3
action_dim = 3
hidden_dim = 256
actor_1 = networks_a2c_v1_2.ActorNetwork_1(state_dim, action_dim, hidden_dim)
critic_1 = networks_a2c_v1_2.CriticNetwork_1(state_dim, hidden_dim)

lr_sheduler_flag = True
warmup_epochs = int(num_iterations*0.05)
lr_rate = 1e-6
#model.load_newModel(actor_1, critic_1)
model.load_newModel(actor_1, critic_1, lr_sheduler_flag, warmup_epochs, num_iterations, lr_rate=lr_rate)


# load model
if testing_exe :

    # Count all files in a directory
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/actor_v1_2'    
    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)

    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints'
    actor_path = folder_path + '/actor_v1_2/checkpoint_episode_' + str(num_files) + '.pt'
    critic_path = folder_path +  '/critic_v1_2/checkpoint_episode_' + str(num_files) + '.pt'

    # IF I WANT A CHECKPOINT
    folder_number = 4
    checkpoint_name = "/checkpoint_episode_33_reward_[327.31729073].pt"

    checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_v1_2'    
    folder_name = checkpoint_path + '/model_v1_2_test_' + str(folder_number)
    
    actor_path = folder_name + '/actor_v1_2' + checkpoint_name
    critic_path = folder_name + '/critic_v1_2' + checkpoint_name

    print()
    print("Model Loaded = ", actor_path)
    print()
    

    model.actor_model, opt_actor, _ = store_model.load_model(model.actor_model, model.opt_actor, actor_path)
    model.critic_model, opt_critic, checkpoint_dict = store_model.load_model(model.critic_model, model.opt_critic, critic_path)

    model.load_optimizers(opt_actor, opt_critic)



for i in range(0, num_iterations):

    # *It shoud be in a reset function*
    states = np.zeros((1, state_dim))
    states_steps = np.zeros((1, state_dim))
    rewards = np.zeros((1, 1))
    rewards_steps = np.zeros((1, 1))
    actions_steps = np.zeros((1, 1))

    done = True
    env.reset_env()
    env.global_iterations = i

    while done :
        
        actions = model.pick_sample(states, vis_flag=False, vis_vals=False)
        env.apply_one_action_left_right(actions)
        env.env_step(normalize_states=True)
        # env.env_step(normalize_states=True, training=not(testing_exe))

        states[0, 0] = env.state_distance[-1][-1]
        states[0, 1] = env.state_dist_to_guideline[-1][-1]
        states[0, 2] = env.state_orientation[-1][-1]

        rewards[0, 0] = env.reward_total_list[-1]
        done = not(env.stop_steps)

        # Store batch
        states_steps = np.vstack( (states_steps, states) )
        rewards_steps = np.vstack( (rewards_steps, rewards) )
        actions_steps = np.vstack( (actions_steps, actions) )

        # env.visuzalization()
        if not(env.running_flag):
            print("Stopped by user")
            break

    
    # Remove the first row (init. array with zeros)
    states_steps = np.delete( states_steps, 0, axis=0)
    rewards_steps = np.delete( rewards_steps, 0, axis=0)
    actions_steps = np.delete( actions_steps, 0, axis=0)
    # print()
    # print(states_steps)
    # print(rewards_steps)
    # print(actions_steps)

    if not testing_exe :
        model.training_a2c(states_steps, actions_steps, rewards_steps, vis_flag=False, clip_grad_flag=True, back_grad_mean=False)
    


    if not(env.running_flag):
        print("Stopped by user")
        break
        


# model.plot_rewards()
model.plot_training(episodes=num_iterations, steps=env.max_steps)



# Saved the last model
print()
if (not testing_exe) :
    store_flag = input("Do you wanna Store the Model? y/n ... ")
else:
    store_flag = 'n'

if (store_flag == 'y') :

    # Count all files in a directory
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/actor_v1_2'    
    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)
    num_name = num_files + 1 

    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints'
    actor_path = folder_path + '/actor_v1_2'
    critic_path = folder_path +  '/critic_v1_2'
    
    store_model.save_model(model.actor_model, model.opt_actor, num_name, rewards_steps, actor_path)
    store_model.save_model(model.critic_model, model.opt_critic, num_name, rewards_steps, critic_path)
    
    print("Saved as = ", actor_path + "_" + str(num_name) )


sys.exit()