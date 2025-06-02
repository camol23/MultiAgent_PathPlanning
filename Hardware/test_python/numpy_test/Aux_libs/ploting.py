import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.patches import RegularPolygon
import matplotlib.animation as animation

colors_list = [
    'deepskyblue',
    'sandybrown',
    'limegreen',
    'gold',
    'orchid',
    'mediumpurple',
    'orange',
    'peru',
    'darkturquoise',
    'coral'
]

color_agent = [
    'steelblue'
]

# https://matplotlib.org/stable/gallery/color/named_colors.html


def plot_scene(route_agent_id, data_lists=[], obst_list=[], target_routes=[[]], cm_flag=False, states=[], obs_unknowns= [], smaller=False):
    '''
        cm_flag: False when map is maller than 20x20 
    '''
    if cm_flag :
        tri_size = 3
        delta_tri = 2      
        delta_name = (5, 2)
    else:
        tri_size = 0.3
        delta_tri = 0.2                # up
        delta_name = (0.5, 0.2)
        # tri_size = 7
        # delta_tri = 6
        # delta_name = (20, 10)

    if smaller :
        tri_size = 0.02
        delta_tri = 0.02                # up
        delta_name = (0.05, 0.02)        


    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    for i, route in enumerate(data_lists) :
        axes.plot(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], label="Agent "+str(route_agent_id[i]) )
        axes.scatter(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], alpha=0.5, linewidths=0.5)

        # Text Names
        axes.text(route[0, 0]-delta_name[0], route[1, 0]-delta_name[1], "Init", fontsize=8, horizontalalignment='left',
        # axes.text(route[0][0]-delta_name[0], route[1][0]-delta_name[1], "Init", fontsize=8, horizontalalignment='left',
                    verticalalignment='center', color=mcolors.CSS4_COLORS['grey'])
        
        if len(states) > 0 :
            axes.plot(states[:, i, 0], states[:, i, 1], color=color_agent[0])

    # Draw the Stop-Goals
    color_tri = mcolors.CSS4_COLORS['lightcoral']
    for i, route in enumerate(target_routes) :        
        for goal_coor in route:
            axes.add_patch(RegularPolygon((goal_coor[0], goal_coor[1]+delta_tri), 3, radius=tri_size, orientation=math.radians(-45), facecolor=color_tri) )

    # Draw obstacles
    num_obs = len(obst_list)
    obst_list = obst_list + obs_unknowns
    # print("OBST = ", obst_list)

    for i in range(0, len(obst_list)):
        # Came from PSO then (x_botton_left, y_botton_left, x_rigth_up, y_rigth_up)
        rect_w = obst_list[i][2]
        rect_w = abs(rect_w - obst_list[i][0])

        rect_h = obst_list[i][3]
        rect_h = abs(rect_h - obst_list[i][1])

        x_botton = obst_list[i][0]
        y_botton = obst_list[i][1]
        print("Obst i ", i, x_botton, y_botton, rect_w, rect_h)
        if i >= num_obs:
            color_obs = 'lightsteelblue'
        else:
            color_obs = 'dimgray'

        axes.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor=mcolors.CSS4_COLORS[color_obs]))

    # Adjust layout
    axes.grid(True)
    axes.legend()

    plt.tight_layout()
    plt.show()





def animate_scene(route_agent_id, data_lists=[], obst_list=[], target_routes=[[]], cm_flag=False, states=[], obs_unknowns=[], smaller=False):
    '''
        
    '''
    if cm_flag :
        tri_size = 3
        delta_tri = 2      
        delta_name = (5, 2)
    else:
        tri_size = 0.3
        delta_tri = 0.2                # up
        delta_name = (0.5, 0.2)
        # tri_size = 7
        # delta_tri = 6
        # delta_name = (20, 10)

    if smaller :
            tri_size = 0.02
            delta_tri = 0.02                # up
            delta_name = (0.05, 0.02)        

    scene_axes = []
    state_lines = [] 
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))    

    # Scene
    for i, route in enumerate(data_lists) :
        scene_axes.append( axes.plot(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], label="Agent "+str(route_agent_id[i]) ) )
        scene_axes.append( axes.scatter(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], alpha=0.5, linewidths=0.5) )

        # Text Names
        scene_axes.append( axes.text(route[0, 0]-delta_name[0], route[1, 0]-delta_name[1], "Init", fontsize=8, horizontalalignment='left',
        # scene_axes.append( axes.text(route[0][0]-delta_name[0], route[1][0]-delta_name[1], "Init", fontsize=8, horizontalalignment='left',
                    verticalalignment='center', color=mcolors.CSS4_COLORS['grey']) )
        
        # state_lines.append( axes.plot(states[0, i, 0], states[0, i, 1], color=color_agent[0]) )

        
    # Draw the Stop-Goals
    color_tri = mcolors.CSS4_COLORS['lightcoral']
    for i, route in enumerate(target_routes) :        
        for goal_coor in route:
            scene_axes.append( axes.add_patch(RegularPolygon((goal_coor[0], goal_coor[1]+delta_tri), 3, radius=tri_size, orientation=math.radians(-45), facecolor=color_tri) ) )

    # Draw obstacles
    num_obs = len(obst_list)
    obst_list = obst_list + obs_unknowns

    for i in range(0, len(obst_list)):
        rect_w = obst_list[i][2]
        rect_w = abs(rect_w - obst_list[i][0])

        rect_h = obst_list[i][3]
        rect_h = abs(rect_h - obst_list[i][1])

        x_botton = obst_list[i][0]
        y_botton = obst_list[i][1]

        if i >= num_obs:
            color_obs = 'lightsteelblue'
        else:
            color_obs = 'dimgray'

        axes.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor=mcolors.CSS4_COLORS[color_obs]))




    # Adjust layout
    # ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
    axes.grid(True)
    # axes.legend()

    return scene_axes, state_lines, fig, axes



def plot_general(data_lists=[], titles=[], num_rows=3):
    '''
        
    '''

    num_plots = len(data_lists)
    num_colmns = math.ceil( num_plots/num_rows )

    print("num_plots = ", num_plots)
    print("num_colmns = ", num_colmns)
    
    counter = 0
    count_colors = 0
    colors = ['r', 'b', 'g', 'y']
    
    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_colmns, figsize=(24, 8))

    for row in range(0, num_rows) :
        for colm in range(0, num_colmns) :
            axes[row, colm].plot(data_lists[counter], colors[count_colors])
            axes[row, colm].set_title( titles[counter] )

            counter += 1
            if count_colors == 3 :
                count_colors = 0
            else:
                count_colors += 1
    

    # Adjust layout
    for axe in axes:
        for ax in axe :
            ax.grid(True)

    plt.tight_layout()
    plt.show()



# Slower down the computation dramatically
class scene_plotter:
    def __init__(self):

        self.figure = None
        self.axe = None
        self.axe_limits = None

        # Params
        self.realtime_flag = None
        self.x_limit = None
        self.y_limit = None
        self.cm_flag = None

        self.tri_size = None
        self.delta_tri = None
        self.delta_name = None

        # Auxiliar
        self.counter = None

    def initialize(self, plot_params):
        
        self.realtime_flag = plot_params['realtime_flag']
        self.x_limit, self.y_limit = plot_params['map_size']        
        self.cm_flag = plot_params['cm_flag']

        # Init.
        self.figure = plt.figure(figsize=(10, 6))
        self.axe = self.figure.add_subplot(1, 1, 1)
        # plt.axis([-10, self.x_limit, -10, self.y_limit]) 
        self.axe.grid(True)
        self.axe.axis('equal')
        plt.show(block=False) 

        if self.cm_flag :
            self.tri_size = 3
            self.delta_tri = 2                # up
            self.delta_name = (5, 2)
        else:
            self.tri_size = 7
            self.delta_tri = 6
            self.delta_name = (20, 10)

        self.counter = 0


    def plot_scene(self, route_agent_id, data_lists=[], obst_list=[], target_routes=[[]], states=None):
        '''
            States dim : ((samples, agents, [x,y,t])
        '''
        
        # Draw Routes
        for i, route in enumerate(data_lists) :
            self.axe.plot(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], label="Agent "+str(route_agent_id[i]) )
            self.axe.scatter(route[0], route[1], color= mcolors.CSS4_COLORS[colors_list[i]], alpha=0.5, linewidths=0.5)

            # Text Names
            self.axe.text(route[0, 0]-self.delta_name[0], route[1, 0]-self.delta_name[1], "Init", fontsize=8, horizontalalignment='left',
                        verticalalignment='center', color=mcolors.CSS4_COLORS['grey'])
            

            # for i in range(0, states.shape[1]):
                # self.axe.plot(states[0:self.counter, i, 0], states[0:self.counter, i, 1], color=color_agent[0])
            self.axe.plot(states[0:self.counter, i, 0], states[0:self.counter, i, 1], color=color_agent[0])

        if self.counter > 1 :    
            print("states[0:self.counter, i, 0] = ", states[0:self.counter, i, 0].shape)
            print("states[0:self.counter, i, 0] = ", states[0:self.counter, i, 0][-1], states[0:self.counter, i, 1][-1])
            print()
        
        # Draw the Stop-Goals
        color_tri = mcolors.CSS4_COLORS['lightcoral']
        for i, route in enumerate(target_routes) :        
            for goal_coor in route:
                self.axe.add_patch(RegularPolygon((goal_coor[0], goal_coor[1]+self.delta_tri), 3, radius=self.tri_size, orientation=math.radians(-45), facecolor=color_tri) )

        # Draw obstacles
        for i in range(0, len(obst_list)):
            rect_w = obst_list[i][2]
            rect_w = abs(rect_w - obst_list[i][0])

            rect_h = obst_list[i][3]
            rect_h = abs(rect_h - obst_list[i][1])

            x_botton = obst_list[i][0]
            y_botton = obst_list[i][1]

            self.axe.add_patch(Rectangle((x_botton, y_botton), rect_w, rect_h, facecolor=mcolors.CSS4_COLORS['dimgray']))

        self.counter = self.counter + 1
        # Adjust layout
        plt.axis([-10, self.x_limit, -10, self.y_limit]) 
        self.axe.grid(True)
        self.axe.axis('equal')
        # self.axe.legend()

        plt.tight_layout()
        plt.show(block=False) 

        plt.draw()
        plt.pause(0.00001)

        
        
def convert_obs_coor(obs_list):
    '''
        input  = (x_botton, y_botton, rect_w, rect_h)
        output = (x_botton_left, y_botton_left, x_rigth_up, y_rigth_up)
    '''
    
    obs_list_output = []
    for i in range(0, len(obs_list)):
        rect_w = obs_list[i][0] + obs_list[i][2]    # x
        rect_h = obs_list[i][1] + obs_list[i][3]    # y
        
        obs_list_output.append((obs_list[i][0], obs_list[i][1], rect_w, rect_h)) 
        
    return obs_list_output