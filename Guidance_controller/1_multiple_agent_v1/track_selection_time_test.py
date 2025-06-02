import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PSO import PSO_decision
from PSO import pso_iterator
from Env import env_small


# Setting params
num_particles = [30, 30, 30, 100]
iterations = [20, 50, 100, 100]
# num_agents = 50
num_test_points = 21


number_obst = 5
max_rect_obs_size = 30

map_size = (250, 250)
# PSO Decision Making
pso_params_routes = {
    'iterations': 20, 
    'w': 0.04, # 0.04
    'Cp': 0.1, #0.2, # 0.7
    'Cg': 0.6, # 0.1 # 0.1
    'num_particles': 30
}

safe_margin_obs_pso = 10 
pso_params = {
    'iterations': 100,  #100 # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': 5 #12 # 5
}


# Storage
num_agents_list = [2*i for i in range(1, num_test_points)]
distance_cost_pso = np.zeros((len(iterations), len(num_agents_list)) )
distance_cost_ref = np.zeros((1, len(num_agents_list)) )

time_pso_list = np.zeros((len(iterations), len(num_agents_list)) )
time_ref_list = np.zeros((1, len(num_agents_list)) )

# Input
# dim = num_test_points - 1
# dist_matrix = np.random.rand(dim, dim)*200
# dim_row = num_test_points - 1  # Num agenst
# dim_row = 3


# number of points (x-axis)
for n_point, num_agents in enumerate(num_agents_list):

    # ----------------------- Creating Input Data -----------------------
    dim_row = num_agents
    dim_colm = 2
    init_pos = np.random.rand(dim_row, dim_colm)*100
    target_pos = (np.random.rand(dim_row, 1, dim_colm)+1)*100

    
    # print("max init ", np.max(init_pos))
    # print("range target ", np.min(target_pos), np.max(target_pos))

    init_agent = init_pos.tolist()
    target_routes = target_pos.tolist()
    # print(init_agent)

    obst_list = env_small.random_obstacles(number_obst, map_size[0], map_size[1], max_rect_obs_size)


    dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function
    an_possible_paths = []                                      # All the computed paths
    data_lists = []                                             # Selected Agent Paths

    pso_routes_time = np.zeros((len(init_agent), 1))
    # Agent n
    for i in range(0, len(init_agent)):

        start_time = time.time()
        pso_iter_an = pso_iterator.PSO_wrapper()
        pso_iter_an.initialization(map_size, init_agent[i], target_routes, pso_params, obst_list)
        pso_iter_an.safe_margin_obs = safe_margin_obs_pso
        pso_iter_an.itereration()
        
        end_time = time.time() - start_time
        pso_routes_time[i, 0] = end_time

        # print("agent ", i, " Dist. ", pso_iter_an.dist_list)
        # print()

        # Stack Routes
        dist_an = np.array(pso_iter_an.dist_list).reshape((len(pso_iter_an.dist_list), 1))    
        dist_matrix = np.hstack((dist_matrix, dist_an))

        # Store possible Paths
        an_possible_paths.append( pso_iter_an.paths )



    dist_matrix = np.delete(dist_matrix, 0, axis=1)
    print("dist_matrix")
    # print(dist_matrix)
    print(dist_matrix.shape)

    # -------------------- Decision --------------------

    for j in range(0, len(num_particles)):

        pso_params_routes['num_particles'] = num_particles[j]
        pso_params_routes['iterations'] = iterations[j]
        start_time = time.time()
        pso_routes = PSO_decision.PSO()
        pso_routes.initialization(pso_params_routes, dist_matrix)
        pso_routes.pso_compute()

        pso_decision_time = time.time() - start_time
        # print("PSO Route")
        # print(pso_routes.output_list)
        # print(pso_routes.output_routes_ids)
        # print("Dist. Cost= ", pso_routes.total_dist([pso_routes.output_list]))
        # print()
        distance_cost_pso[j, n_point] = pso_routes.total_dist([pso_routes.output_list]).item()
        time_pso_list[j, n_point] = pso_decision_time

        # Checking ...
        # print("Reference sol. ")
        # print(pso_routes.ref_solution)
        # print("Dist. Cost= ", pso_routes.total_dist([pso_routes.ref_solution]))
        distance_cost_ref[0, n_point] = pso_routes.total_dist([pso_routes.ref_solution]).item()
        time_ref_list[0, n_point] = pso_routes.time_ref_path




# Ploting
colors = [
    'rosybrown',
    'steelblue',
    'mediumpurple',
    'bisque',
    'darkseagreen'
]

# fig = plt.figure() 
# ax = fig.add_subplot(1, 1, 1) 
fig, axes = plt.subplots(1, 2)

y_axis_pso_list = [time_pso_list*1000, 100*(1-(distance_cost_pso/distance_cost_ref))]
y_axis_ref_list = [time_ref_list*1000, 100*(1-(distance_cost_ref/distance_cost_ref))]
ylabel_list = ['Time [ms]', 'Cost reduction respect Deterministic [%]']

for j in range(0, len(y_axis_pso_list)):

    y_axis_pso = y_axis_pso_list[j]
    y_axis_ref = y_axis_ref_list[j]

    for i, num_iterations in enumerate(iterations):

        # color_i = mcolors.CSS4_COLORS[ploting.colors_agent[i]]
        color_i = mcolors.CSS4_COLORS[colors[i]]
        axes[j].plot(num_agents_list, y_axis_pso[i, :], color =color_i, alpha=0.7, label="PSO Iter " +str(num_iterations) + " p " + str(num_particles[i]))
        axes[j].scatter(num_agents_list, y_axis_pso[i, :], color=color_i, alpha=0.5, linewidths=0.5)


    color_i = mcolors.CSS4_COLORS[colors[-1]]
    axes[j].plot(num_agents_list, y_axis_ref[0, :], color =color_i, alpha=0.7, label="Deterministic")
    axes[j].scatter(num_agents_list, y_axis_ref[0, :], color=color_i, alpha=0.5, linewidths=0.5)
        
    axes[j].grid(which="major", color="0.9")
    axes[j].legend() #loc=2
    axes[j].set_xlabel('Number of agents')
    axes[j].set_ylabel(ylabel_list[j])
# axes.grid(True)
# ax.axis('equal')
plt.show()         



# fig, axes = plt.subplots(1, 1)

# for i, num_iterations in enumerate(iterations):

#     # color_i = mcolors.CSS4_COLORS[ploting.colors_agent[i]]
#     color_i = mcolors.CSS4_COLORS[colors[i]]
#     axes.plot(num_agents_list, time_pso_list[i, :], color =color_i, alpha=0.7, label="PSO Iter " +str(num_iterations) + " p " + str(num_particles[i]))
#     axes.scatter(num_agents_list, time_pso_list[i, :], color=color_i, alpha=0.5, linewidths=0.5)


# color_i = mcolors.CSS4_COLORS[colors[-1]]
# axes.plot(num_agents_list, time_ref_list[0, :], color =color_i, alpha=0.7, label="Deterministic")
# axes.scatter(num_agents_list, time_ref_list[0, :], color=color_i, alpha=0.5, linewidths=0.5)
    
# axes.grid(which="major", color="0.9")
# axes.legend() #loc=2
# axes.set_xlabel('Number of agents')
# axes.set_ylabel('Time [ms]')
# # axes.grid(True)
# # ax.axis('equal')
# plt.show()         


