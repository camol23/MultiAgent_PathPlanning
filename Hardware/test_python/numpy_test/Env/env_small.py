import math






def load_scene(load_scene_params):
    '''
        Define:
            (1) Init Pos
            (2) Target    
            (3) Number of Agents
            (4) Load Map
                4.1) Obstacles
                4.2) obst_list_unkowns
                4.3) Map size   

    '''
    scene_name = load_scene_params['scene_name']

    if scene_name == 'scene_0_a5' :
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_0' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)
    
    elif scene_name == 'scene_obs_0_a5' :
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_obs_0' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_0_a2' :
        init_agent = [(10, 100), (10, 130)]
        x_end = 200
        target_routes = [                 
                 [(x_end, 190)],                # Route 1
                 [(x_end, 210)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_0' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)


    elif scene_name == 'scene_force_collision_obs' :
        init_agent = [(10, 50), (10, 80)]
        x_end = 190
        target_routes = [                 
                 [(x_end, 160)],                # Route 1
                 [(x_end, 190)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_force_collision_obs' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)
    
    
    elif scene_name == 'scene_force_collision_obs_a1' :
        init_agent = [(40, 140), (100, 180)]
        target_routes = [                 
                 [(55, 75)],                # Route 1
                 [(110, 1)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_force_collision_obs' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_no_obst' :
        # init_agent = [(10, 50), (10, 80)]
        init_agent = [(10, 160), (10, 190)]
        x_end = 190
        target_routes = [                 
                 [(x_end, 160)],                # Route 1
                 [(x_end, 190)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_no_obst' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)
    
    elif scene_name == 'scene_obs_0_a5_complement_1' :
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_obs_0_complement_1' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    
    elif scene_name == 'scene_experiment_test0_a2' :
        # Straight line
        init_agent = [(0, 0), (0, 80)]
        x_end = 100
        target_routes = [
                 [(x_end, 0)],                 # Route 1                 
                 [(x_end, 80)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_no_obst' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    else:
        # Default: scene map_0_a5
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_0' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)
    


    return init_agent, target_routes, obst_list, obst_list_unkowns, map_size


def load_maps(load_map_params):
    '''
        Obstacle : (x_botton_left, y_botton_left, width, heigth)
    
    '''
    
    map_name = load_map_params['map_name']

    map_size = (250, 250)
    obst_list = []
    obst_list_unkowns = []
    if map_name == 'map_0' :
        map_size = (250, 250)
        #  if rigth corner is smaller then plays as width and heigth
        obst_list.append((130, 125, 20, 40))
        obst_list.append((70, 50, 20, 40)) 

    elif map_name == 'map_obs_0' :
        map_size = (250, 250)
        
        obst_list.append((130, 125, 20, 40))
        obst_list.append((50, 25, 20, 40))      
        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns = [(75, 125, 20, 40)]
    
    elif map_name == 'map_force_collision_obs' :
        map_size = (200, 200)
        obst_list_unkowns.append((50, 50, 150, 150))

    elif map_name == 'map_no_obst' :
        map_size = (200, 200)       

    elif map_name == 'map_obs_0_complement_1' :
        map_size = (250, 250)        
        obst_list = [(115, 10, 20, 40)]
       

    else: # Default is map_0
        map_size = (250, 250)
        obst_list.append((70, 50, 20, 40))


    return obst_list, obst_list_unkowns, map_size 



def create_maps():
    '''
        Randomize the Obstacles
    '''
    pass
    