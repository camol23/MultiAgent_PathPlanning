import sys
import numpy as np

from Env import env_v1
from PSO import PSO_v1
from PSO import PSO_v2
from Env.agents_v1 import follow_path_wp 



# Agents Settings
agents_settings = {
               # (x0, y0)
    'start_pos': (50, 550),
    'num_agents': 3,
    'formation_type': 2
}

# Map Settings
map_settings = {
    'map_dimensions': (1200, 600),
    'num_obs': 30,
    'type_obs': 'random',                  # Simple Map Grid
    # 'type_obs': 'warehouse_0',                  # More elements Map Grid
    'max_rect_obs_size': 200,                   # maximun Obstacle size
    'seed_val_obs': 80, # 286                   # Test obstacles location
    'mouse_flag': True                          # Mouse pointer is turned in a sqaere obstacle
}

# 860

# PSO Settings
pso_params = {
    'iterations': 200, 
    'w': 0.04, # 0.04
    'Cp': 0.2, #0.2,
    'Cg': 0.1,
    'num_particles': 100,
    'resolution': 10
}




# Initialize Environment
env = env_v1.Environment(map_settings, agents_settings)
env.initialize()


# Compute the PSO Path 
target_pos = (1100, 100)

# pso_item = PSO_v1.PSO(map_settings['map_dimensions'], agents_settings['start_pos'], target_pos, pso_params, env.env_map.random_rect_obs_list)
pso_item = PSO_v2.PSO(map_settings['map_dimensions'], agents_settings['start_pos'], target_pos, pso_params, env.env_map.random_rect_obs_list)
pso_item.pso_compute()


pso_item.visualization()

# pso_item.collision_rect_lastCorrection(pso_item.G[0, :])
pso_item.visualization_lastAdjustment()
pso_item.visualization_all()

# env.env_map.path_agent = np.copy(pso_item.output_path)
print("Shape PSO Path", pso_item.output_path.shape)
env.load_path(pso_item.output_path)


# Straitgh Line (for test with no PSO)
# path = np.transpose(np.array([agents_settings['start_pos'], target_pos]))
# env.load_path(path)


while env.running_flag:

    env.env_step()

    # Policy Test (Action generation)
    for agent in env.agents_obj :
        agent.heading, agent.wp_current, stop_signal = follow_path_wp(agent, env.reference_path)
        
        if stop_signal:
            agent.move_stop()


sys.exit()