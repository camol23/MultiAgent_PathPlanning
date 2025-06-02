import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
from matplotlib.animation import FuncAnimation

import numpy as np
import math

class RealtimePlot:
    '''
        max_points: Change the number of points shown at once
        interval: Adjust the animation update interval (in milliseconds)
        ylim: Modify the initial y-axis limits in the constructor
    '''
    def __init__(self, max_points=100):
        # Create figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.max_points = max_points
        
        # Initialize empty lists for data
        self.x_data = list(range(max_points))
        self.y_data = [0] * max_points
        
        # Create initial line
        self.line, = self.ax.plot(self.x_data, self.y_data, 'b-')
        
        # Set up the plot
        self.ax.set_xlim(0, max_points)
        self.ax.set_ylim(-1, 1)  # Adjust as needed
        self.ax.grid(True)
        self.ax.set_title('Real-time Plot')
        
        # Show the plot
        plt.show()
        
    def add_point(self, value):
        # Add new data point
        self.y_data.pop(0)
        self.y_data.append(value)
        
        # Update line data
        self.line.set_ydata(self.y_data)
        
        # Update y-axis limits if needed
        ymin, ymax = min(self.y_data), max(self.y_data)
        margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1  # 10% margin
        self.ax.set_ylim(ymin - margin, ymax + margin)
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage
# if __name__ == "__main__":
#     import time
    
#     # Create the real-time plot
#     plotter = RealtimePlot(max_points=50)
    
#     # Simulate data generation and plotting
#     try:
#         for i in range(200):
#             # Generate some sample data (replace with your data source)
#             value = np.sin(i * 0.1) + np.random.normal(0, 0.1)
            
#             # Add the new point to the plot
#             plotter.add_point(value)
            
#             # Your other code can go here
#             print(f"Processing iteration {i}")
            
#             # Small pause to make the animation visible
#             time.sleep(0.05)
            
#     except KeyboardInterrupt:
#         print("Stopped by user")
    
#     # Keep the plot window open
#     plt.ioff()
#     plt.show()



class MultiRealtimePlot:
    '''
        Number of plots by changing num_plots
        Plot titles by modifying the titles list
        Colors by modifying the colors list
        Figure size by adjusting figsize
        Individual plot properties through self.axes[i]
    '''
    def __init__(self, num_plots=3, max_points=100):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, num_plots, figsize=(15, 5))
        self.max_points = max_points
        self.num_plots = num_plots
        
        # Initialize data for each plot
        self.x_data = list(range(max_points))
        self.y_data = [[0] * max_points for _ in range(num_plots)]
        self.lines = []
        
        # Set up each subplot
        titles = ['Plot 1', 'Plot 2', 'Plot 3']
        colors = ['b-', 'r-', 'g-']
        
        for i in range(num_plots):
            line, = self.axes[i].plot(self.x_data, self.y_data[i], colors[i])
            self.lines.append(line)
            
            self.axes[i].set_xlim(0, max_points)
            self.axes[i].set_ylim(-1, 1)
            self.axes[i].grid(True)
            self.axes[i].set_title(titles[i])
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
    def add_points(self, values):
        """
        Add new points to all plots
        values: list of values, one for each plot
        """
        if len(values) != self.num_plots:
            raise ValueError(f"Expected {self.num_plots} values, got {len(values)}")
            
        for i in range(self.num_plots):
            # Update data for each plot
            self.y_data[i].pop(0)
            self.y_data[i].append(values[i])
            
            # Update line data
            self.lines[i].set_ydata(self.y_data[i])
            
            # Update y-axis limits
            ymin, ymax = min(self.y_data[i]), max(self.y_data[i])
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
            self.axes[i].set_ylim(ymin - margin, ymax + margin)
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage
# if __name__ == "__main__":
#     import time
    
#     # Create the real-time plots
#     plotter = MultiRealtimePlot(num_plots=3, max_points=50)
    
#     try:
#         for i in range(200):
#             # Generate different sample data for each plot
#             value1 = np.sin(i * 0.1) + np.random.normal(0, 0.1)  # Sine wave + noise
#             value2 = np.cos(i * 0.1) + np.random.normal(0, 0.1)  # Cosine wave + noise
#             value3 = np.sin(i * 0.05) * np.cos(i * 0.1)  # More complex pattern
            
#             # Add new points to all plots
#             plotter.add_points([value1, value2, value3])
            
#             # print(f"Processing iteration {i}")
#             time.sleep(0.05)
            
#     except KeyboardInterrupt:
#         print("Stopped by user")
    
#     plt.ioff()
#     plt.show()


def plot_list(list_1, list_2, list_3, titles):
        '''
            Plot in a row

                Args: 
                    (1) Lists
                    (2) titles := List of Strings 
                
        '''
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # First plot
        axes[0].plot(list_1, 'r')
        axes[0].set_title(titles[0])

        # Second plot
        axes[1].plot(list_2, 'g')
        axes[1].set_title(titles[1])

        # Third plot
        axes[2].plot(list_3, 'b')
        axes[2].set_title(titles[2])


        # Adjust layout
        for ax in axes:
            # ax.set_aspect('equal')
            ax.grid(True)

        plt.tight_layout()
        plt.show()



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