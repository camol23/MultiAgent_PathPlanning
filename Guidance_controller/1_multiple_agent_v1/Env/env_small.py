import math
import random

# Env. of Control

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

    elif scene_name == 'scene_obs_2_a5' :
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_obs_2' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_obs_3_a5' :
        init_agent = [(10, 10), (10, 40), (10, 70), (10, 100), (10, 130)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_obs_3' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_2_obs_5_a5' :
        init_agent = [(5, 40), (5, 70), (5, 100), (5, 130), (5, 160)]
        x_end = 200
        target_routes = [
                 [(175, 25), (x_end, 100)],     # Route 1
                 [(x_end, 130)],                # Route 2
                 [(x_end, 160)],                # Route 3
                 [(x_end, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_2_obs_5' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_2_obs_7_a5' :
        init_agent = [(5, 40), (5, 70), (5, 100), (5, 130), (5, 160)]
        x_end = 200
        target_routes = [
                 [(190, 50), (x_end+15, 100)],     # Route 1
                 [(x_end+11, 130)],                # Route 2
                 [(x_end+6, 160)],                # Route 3
                 [(x_end+2, 190)],                # Route 4
                 [(x_end, 210)] ]               # Route 5

        load_map_params = { 'map_name' : 'map_2_obs_7' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_0_a2' :
        init_agent = [(10, 100), (10, 130)]
        x_end = 200
        target_routes = [                 
                 [(x_end, 190)],                # Route 1
                 [(x_end, 210)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_0' }
        obst_list, obst_list_unkowns, map_size = load_maps(load_map_params)

    elif scene_name == 'scene_0_a2_exp_line' :
        init_agent = [(0, 10), (0, 60)]
        x_end = 100
        target_routes = [                 
                 [(x_end, 10)],                # Route 1
                 [(x_end, 60)] ]               # Route 2

        # map (200, 200)
        load_map_params = { 'map_name' : 'map_no_obst' }
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
        init_agent = [(0, 0), (0, 60)]

        target_routes = [
                 [(175, 100)],                 # Route 1                 
                 [(150, 150)] ]               # Route 2

        load_map_params = { 'map_name' : 'map_experiment_test0_a2' }
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
        map_size = (225, 225)
        #  if rigth corner is smaller then plays as width and heigth
        obst_list.append((130, 125, 20, 40))
        obst_list.append((70, 50, 20, 40)) 

    elif map_name == 'map_obs_0' :
        map_size = (225, 225)
        
        obst_list.append((130, 125, 20, 40))
        obst_list.append((50, 25, 20, 40))      
        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns = [(75, 125, 20, 40)]
    
    elif map_name == 'map_obs_2' :
        map_size = (225, 225)
        
        obst_list.append((130, 125, 20, 40))
        obst_list.append((50, 25, 20, 40))      
        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns.append((75, 125, 20, 40))
        obst_list_unkowns.append((100, 0, 50, 50))
    
    elif map_name == 'map_obs_3' :
        map_size = (225, 225)
        
        obst_list.append((145, 130, 20, 40))
        obst_list.append((50, 25, 20, 40))      
        obst_list.append((0, 150, 75, 75))      

        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns.append((75, 125, 20, 40))
        obst_list_unkowns.append((100, 0, 50, 50))
        obst_list_unkowns.append((100, 200, 25, 25))

    elif map_name == 'map_2_obs_5' :
        map_size = (225, 225)
        
        obst_list.append((40, 40, 10, 50))
        obst_list.append((50, 130, 10, 40))      
        obst_list.append((130, 140, 40, 30))      

        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns.append((70, 0, 20, 20))
        obst_list_unkowns.append((130, 0, 20, 20))
        obst_list_unkowns.append((100, 55, 20, 20))
        obst_list_unkowns.append((75, 140, 20, 20))
        obst_list_unkowns.append((100, 200, 20, 20))

    elif map_name == 'map_2_obs_7' :
        map_size = (225, 225)
        
        obst_list.append((40, 40, 30, 50))
        obst_list.append((30, 120, 10, 20))      
        obst_list.append((125, 140, 30, 15))      

        # obst_list_unkowns = [(75, 125, 95, 165)]
        obst_list_unkowns.append((90, 0, 20, 20))
        obst_list_unkowns.append((140, 0, 20, 20))
        obst_list_unkowns.append((100, 55, 20, 20))
        obst_list_unkowns.append((150, 75, 10, 10))
        obst_list_unkowns.append((75, 125, 20, 20))
        obst_list_unkowns.append((70, 200, 20, 20))
        obst_list_unkowns.append((130, 200, 20, 20))
        
    
    elif map_name == 'map_force_collision_obs' :
        map_size = (200, 200)
        obst_list_unkowns.append((50, 50, 100, 100))

    elif map_name == 'map_no_obst' :
        map_size = (200, 200)       

    elif map_name == 'map_obs_0_complement_1' :
        map_size = (250, 250)        
        obst_list = [(115, 10, 20, 40)]

    elif map_name == 'map_experiment_test0_a2' :
        map_size = (200, 200)
        
        obst_list.append((100, 100, 36, 20))              # MasterChef
        obst_list.append((41, 20, 42, 31))            # Lego
        obst_list_unkowns = [(45, 120, 21, 17)]         # YouSee
       

    else: # Default is map_0
        map_size = (250, 250)
        obst_list.append((70, 50, 20, 40))


    return obst_list, obst_list_unkowns, map_size 



def create_maps():
    '''
        Randomize the Obstacles
    '''
    pass


def random_obstacles(number, map_width, map_height, max_rect_obs_size, seed_val = None, vis=False ):
    ''' 
        
    '''        

    # print('seed val = ', seed_val)
    if seed_val != None:
        random.seed(seed_val)

    random_rect_obs_list = []
    for i in range(0, number):
        scale_width = random.uniform(0.1, 1)
        scale_height = random.uniform(0.1, 1)
        x_factor = random.uniform(0.1, 1)
        y_factor = random.uniform(0.1, 1)

        # x_rect = min(max(x_factor*map_width, 20), map_width-20)
        y_rect = y_factor*map_height
        rect_w = scale_width*max_rect_obs_size
        rect_h = scale_height*max_rect_obs_size
        x_rect = min(max(x_factor*map_width, 20), map_width-20-rect_w)

        rect_w_aux = rect_w + x_rect
        if rect_w_aux > map_width :
            rect_w = map_width - x_rect
        
        rect_h_aux = rect_h + y_rect
        if rect_h_aux > map_height :
            rect_h = map_height - y_rect
        
        random_rect_obs_list.append( (int(x_rect), 
                                            int(y_rect), 
                                            int(rect_w), 
                                            int(rect_h)) )
        
        if vis :
            print("Obst " + str(i) + " = ", random_rect_obs_list[i])

    return random_rect_obs_list