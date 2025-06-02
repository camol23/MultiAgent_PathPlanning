import numpy as np

from PSO import pso_iterator
from PSO import PSO_decision
from Aux_libs import ploting

# Settings
Plot_from_PSO = False

# map
map_size = (1200, 600)

# PSO Settings
pso_params = {
    'iterations': 100,  # 200
    'w': 0.04, # 0.04
    'Cp': 0.4, #0.2, # 0.7
    'Cg': 0.2, # 0.1 # 0.1
    'num_particles': 100,
    'resolution': 5
}

# PSO Decision Making
pso_params_routes = {
    'iterations': 50, 
    'w': 0.04, # 0.04
    'Cp': 0.1, #0.2, # 0.7
    'Cg': 0.6, # 0.1 # 0.1
    'num_particles': 3
}


# Agents
             # agent 1 ... agent n
init_agent = [(50, 550), (50, 300)]

# Routes
target_routes = [[(600, 500), (1100, 100)],    # Route 1
                 [(900, 100)] ]                # Route 2


# Obstacles
# x_botton, y_botton, width, height
obst_list = []
obst_list.append((400, 200, 200, 200))


#
# PSO Computation
#
# Agent 1
pso_iter_a1 = pso_iterator.PSO_wrapper()
pso_iter_a1.initialization(map_size, init_agent[0], target_routes, pso_params, obst_list)
pso_iter_a1.itereration(plot_flag=Plot_from_PSO)

print("agent 1 Dist. ", pso_iter_a1.dist_list)
print()

# Agent 2
pso_iter_a2 = pso_iterator.PSO_wrapper()
pso_iter_a2.initialization(map_size, init_agent[1], target_routes, pso_params, obst_list)
pso_iter_a2.itereration(plot_flag=Plot_from_PSO)

print("agent 2 Dist. ", pso_iter_a2.dist_list)
print()


# Stack Routes
dist_a1 = np.array(pso_iter_a1.dist_list) 
dist_a2 = np.array(pso_iter_a2.dist_list) 

dist_matrix = np.stack((dist_a1, dist_a2), axis=1)
print("dist_matrix")
print(dist_matrix)


# Syntetic data for dist_Matrix
# dist_matrix = np.array([[2, 5], [6, 3]])

# Decision
agent_list = pso_iterator.assign_paths(dist_matrix)
print()
print("Base Algorithm =", agent_list)





# Test PSO for Assigning routes task
pso_routes = PSO_decision.PSO()
pso_routes.initialization(pso_params_routes, dist_matrix)
pso_routes.pso_compute()

print("PSO Route")
print(pso_routes.output_list)
print(pso_routes.output_routes_ids)
print()


# Checking ...
# print(pso_routes.ref_solution)


# Take the chosen routes
agent_1 = pso_iter_a1.paths[pso_routes.output_routes_ids[0]] 
agent_2 = pso_iter_a2.paths[pso_routes.output_routes_ids[1]] 

data_lists = [agent_1, agent_2]


ploting.plot_scene(pso_routes.output_list, data_lists, pso_iter_a1.obs_rect_list_original, target_routes)