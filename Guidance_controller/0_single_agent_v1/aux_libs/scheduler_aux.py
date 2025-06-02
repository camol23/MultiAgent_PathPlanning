

class scheduler_linear:
    def __init__(self, data):
        
        self.init_val = data['init_val']            # Starting Point
        self.max_val = data['max_steps']            # Maimum Point
        self.val = data['init_val']                 # Output
        
        self.step = 0                               # step delta (linear update)
        self.div_episodes = data['div_episodes']    # Update time
        self.counter = 0                                # COunt the number of episodes
        self.total_episodes = data['num_episodes']
        self.update_episodes = 0                    # how many episodes to update


    def init(self):
        val_span = int( self.max_val - self.init_val )
        self.update_episodes = int( self.total_episodes/self.div_episodes )

        self.step =  int( val_span/self.div_episodes )

        print("self.step = ", self.step)
        print("update_episodes = ", self.update_episodes)
        print("val_span = ", val_span)


    def update(self):
        
        if self.counter >= self.update_episodes :

            if self.val >= self.max_val :
                self.val = self.max_val
            else:
                self.val = self.val + self.step

            self.counter = 0

        self.counter += 1 
