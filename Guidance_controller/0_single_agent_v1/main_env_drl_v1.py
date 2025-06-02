import sys
import numpy as np

from Env import env_v1
from DRL import a2c_test_v1


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

goal_pos = (700, 300) #(1000, 200)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# DRL model
model = a2c_test_v1.drl_model()

# Training Parameters
num_iterations = 1000
env.max_steps = 500



for i in range(0, num_iterations):

    # *It shoud be in a reset function*
    states = np.zeros((1, 1, 2))
    states_steps = np.zeros((1, 1, 2))
    rewards = np.zeros((1, 1))
    rewards_steps = np.zeros((1, 1))
    # actions = np.zeros((1, 1))
    actions_steps = np.zeros((1, 1))

    done = True
    env.reset_env()
    env.global_iterations = i

    while done :
        
        actions = model.pick_sample(states)
        env.apply_actions_left_right(actions)
        env.env_step(normalize_states=True, training=True)

        states[0, 0, 0] = env.state_distance[-1][-1]
        states[0, 0, 1] = env.state_dist_to_guideline[-1][-1]

        rewards[0, 0] = env.reward_total_list[-1]
        done = not(env.stop_steps)

        # Store batch
        states_steps = np.vstack( (states_steps, states) )
        rewards_steps = np.vstack( (rewards_steps, rewards) )
        actions_steps = np.vstack( (actions_steps, actions) )

        env.visuzalization()

    
    # Remove the first row (init. array with zeros)
    states_steps = np.delete( states_steps, 0, axis=0)
    rewards_steps = np.delete( rewards_steps, 0, axis=0)
    actions_steps = np.delete( actions_steps, 0, axis=0)
    # print()
    # print(states_steps)
    # print(rewards_steps)
    # print(actions_steps)

    model.training_a2c(states_steps, actions_steps, rewards_steps)
    


    if not(env.running_flag):
        print("Stopped by user")
        break
        


model.plot_results()
sys.exit()