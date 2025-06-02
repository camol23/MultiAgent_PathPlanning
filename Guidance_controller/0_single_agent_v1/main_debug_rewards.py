import sys
import time
import numpy as np
import math

from Env import env_v1
from aux_libs import ploting


# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (100, 550),  #(50, 550),
    # 'start_pos': (100, 400),
    'num_agents': 1,
    'formation_type': 2                         # 2: V formation
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    'num_obs': 3,
    'type_obs': None,                           # Random sqaure obstacles
    'seed_val_obs': 286,                        # Test obstacles location
    'mouse_flag': True,                          # Mouse pointer is turned in a sqaere obstacle
    'max_rect_obs_size': 200                    # maximun Obstacle size
}

map_training_params = {
    'goal_tolerance' : 0.03,                        # Number of times to reach goal to pass to the next one
    'wait_in_wp'     : 1,                           # Goal zone Margin (around the goal point)
    'section_type' : 'div_segments',              # Slect the WP generation with the Defined path (env.load())
    'path_points'    : 2                            # Num. of WP in the Defined path 
}

# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings, training_flag=True)
env.initialize()

goal_pos = (700, 300) #(1000, 200)
goal_pos = (220, 480)
path = np.transpose(np.array([agents_settings['start_pos'], goal_pos]))
env.load_path(path, map_training_params['section_type'], map_training_params['path_points'])

env.max_steps = math.inf
# env.load_path(path)
print("Goal point = ", path[0, -1], path[1, -1])
print("Start point = ", path[0, -2], path[1, -2])


# plotter = ploting.MultiRealtimePlot(num_plots=3, max_points=50)

while env.running_flag:

    env.env_step(normalize_states=True)
    # env.agents_obj[0].heading = math.radians(-22)

    # time.sleep(0.05)
    # env.visuzalization()

    # print("BREAK")
    # break

    # Debuging States and Rewards functions
    # env.state_angl_between()
    # env.compute_angl_error_reward()

    # env.compute_distance_to_goal()
    # env.compute_distance_reward()


    # Real-Time plot  (It can be slow)
    # value1 = env.state_distance[-1][-1]
    # value2 = env.reward_distance_list[-1][-1]
    # value3 = env.reward_dist_guideline__semiDiscrete_list[-1][-1]

    # plotter.add_points([value1, value2, value3])

    
    

# Plot  
print()
plot_individually = False
plot_flag = input("Do you wanna Plot? y/n ... ")

if plot_flag == 'y' :

    plot_1 = np.array(env.state_distance).squeeze()
    plot_2 = np.array(env.reward_distance_list).squeeze()
    plot_3 = np.array(env.reward_distance_semiDiscrete_list).squeeze()

    plot_4 = np.array(env.reward_total_list).squeeze()
    reward_trajectory = np.sum(plot_4)

    plot_5 = np.array(env.state_dist_to_guideline).squeeze()
    plot_6 = np.array(env.reward_dist_guideline_list).squeeze()
    plot_7 = np.array(env.reward_dist_guideline__semiDiscrete_list).squeeze()

    # plot_8 = np.array(env.state_orientation).squeeze()
    # plot_8 = np.rad2deg(plot_8)
    # plot_9 = np.array(env.reward_orientation_list).squeeze()
    # plot_10 = np.array(env.reward_orientation_attenuated_list).squeeze()

    # Heading
    plot_8 = np.array(env.state_heading).squeeze()
    plot_8 = np.rad2deg(plot_8)
    plot_9 = np.array(env.reward_heading_error).squeeze()

    plot_10 = np.array(env.reward_orientation_list).squeeze()
    plot_11 = np.array(env.theta_goal_heading_test).squeeze()
    plot_11 = np.rad2deg(plot_11)

    titles_all = ['Dist. State', 'Lin. Reward', 'Semi-Discrete Reward', 'Total Reward ' + str(reward_trajectory),
                  'Dist. guide-line State', 'Lin. Reward', 'Semi-Discrete Reward', '',
                  'Heading Angle', 'Lin. Reward Heading', 'Lin. Reward - Orientation', 'theta goal' ]
                #    'Orientation Angle', 'Lin. Reward', 'Attenuated Reward', '' ]

    data_lists = [plot_1, plot_2, plot_3, plot_4,
                  plot_5, plot_6, plot_7, [],
                  plot_8, plot_9, plot_10, plot_11]

    ploting.plot_general(data_lists, titles_all, num_rows=3)




    if plot_individually :
        list_1 = np.array(env.state_distance).squeeze()
        list_2 = np.array(env.reward_distance_list).squeeze()
        list_3 = np.array(env.reward_distance_semiDiscrete_list).squeeze()
        titles = ['Dist. State', 'Lin. Reward', 'Semi-Discrete Reward']

        ploting.plot_list(list_1, list_2, list_3, titles)


        list_1 = np.array(env.state_dist_to_guideline).squeeze()
        list_2 = np.array(env.reward_dist_guideline_list).squeeze()
        list_3 = np.array(env.reward_dist_guideline__semiDiscrete_list).squeeze()
        titles = ['Dist. guide-line State', 'Lin. Reward', 'Semi-Discrete Reward']

        ploting.plot_list(list_1, list_2, list_3, titles)

        list_1 = np.array(env.state_orientation).squeeze()
        list_1 = np.rad2deg(list_1)
        list_2 = np.array(env.reward_orientation_list).squeeze()
        list_3 = np.array(env.reward_orientation_attenuated_list).squeeze()

        titles = ['Orientation Angle', 'Lin. Reward', 'Attenuated Reward']

        ploting.plot_list(list_1, list_2, list_3, titles)




sys.exit()