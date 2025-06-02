import numpy as np

from PSO import pso_iterator
from PSO import PSO_decision
from Aux_libs import ploting

# Settings
Plot_from_PSO = False

# map
map_size = (200, 200)

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
    'iterations': 20, 
    'w': 0.04, # 0.04
    'Cp': 0.1, #0.2, # 0.7
    'Cg': 0.6, # 0.1 # 0.1
    'num_particles': 30
}


# Agents
             # agent 1 ... agent n
init_agent = [(10, 180), (10, 130), (10, 80)]

# Routes
target_routes = [[(120, 150), (185, 100)],    # Route 1
                 [(175, 60)],                # Route 2
                 [(150, 20)] ]               # Route 3


# Obstacles
# x_botton, y_botton, width, height
obst_list = []
# obst_list.append((80, 80, 20, 40))
obst_list.append((50, 120, 20, 40))
obst_list.append((75, 40, 20, 40))
obst_list.append((125, 80, 20, 40))


#
# PSO Computation
#
dist_matrix = np.zeros((len(target_routes), 1))             # Init. the matrix for stack function
an_possible_paths = []                                      # All the computed paths
data_lists = []                                             # Selected Agent Paths

# Agent n
for i in range(0, len(init_agent)):

    pso_iter_an = pso_iterator.PSO_wrapper()
    pso_iter_an.initialization(map_size, init_agent[i], target_routes, pso_params, obst_list)
    pso_iter_an.itereration(plot_flag=Plot_from_PSO)

    print("agent ", i, " Dist. ", pso_iter_an.dist_list)
    print()

    # Stack Routes
    dist_an = np.array(pso_iter_an.dist_list).reshape((len(pso_iter_an.dist_list), 1))    
    dist_matrix = np.hstack((dist_matrix, dist_an))

    # Store possible Paths
    an_possible_paths.append( pso_iter_an.paths )



dist_matrix = np.delete(dist_matrix, 0, axis=1)
print("dist_matrix")
print(dist_matrix)

# Take Obstacle list
obst_original = pso_iter_an.obs_rect_list_original


#
# Decision Making
#
# Test PSO for Assigning routes task
pso_routes = PSO_decision.PSO()
pso_routes.initialization(pso_params_routes, dist_matrix)
pso_routes.pso_compute()

print("PSO Route")
print(pso_routes.output_list)
print(pso_routes.output_routes_ids)
print("Dist. Cost= ", pso_routes.total_dist([pso_routes.output_list]))
print()


# Checking ...
print("Reference sol. ")
print(pso_routes.ref_solution)
print("Dist. Cost= ", pso_routes.total_dist([pso_routes.ref_solution]))


# Take the chosen routes
for i, paths in enumerate(an_possible_paths) :
    selected_path = paths[pso_routes.output_routes_ids[i]]

    data_lists.append( selected_path )


ploting.plot_scene(pso_routes.output_list, data_lists, obst_original, target_routes, cm_flag=True)