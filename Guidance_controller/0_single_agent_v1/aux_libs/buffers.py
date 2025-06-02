import numpy as np
import copy

from collections import deque
import random



# Memory Buffer for storing trajectories
class trajectory_mem:
    def __init__(self):
        
        # save current vals.
        self.state_dim = None
        self.states = None
        self.rewards = None
        self.next_states = None

        self.done = None

        # save trajectory
        self.states_steps = None        
        self.actions_steps = None
        self.log_probs_steps = None
        self.rewards_steps = None
        self.next_states_steps = None

        self.dones_steps = None

        # Axu. Vals.
        self.clear_flag = False                     # To remove first row in 'store_trajectory'
        self.active_states = []                     # List with the names of the States to be read


    def initialize(self, state_dim, active_states=[]):
        
        self.state_dim = state_dim
        self.active_states = active_states 
        self.clear()

    def take_states(self, env):
        '''
            Read the states from Env. Object
        '''
        for state_name in self.active_states :

            if state_name == 'dist' :
                self.next_states[0, 0] = env.state_distance[-1][-1]
            if state_name == 'dist_guideline' : 
                self.next_states[0, 1] = env.state_dist_to_guideline[-1][-1]
            if state_name == 'orientation' :
                self.next_states[0, 2] = env.state_orientation[-1][-1]
            
            if state_name == 'heading' :
                self.next_states[0, 2] = env.state_heading[-1][-1]  # [0, 1]
        
        # self.states[0, 0] = env.state_distance[-1][-1]
        # self.states[0, 1] = env.state_dist_to_guideline[-1][-1]
        # self.states[0, 2] = env.state_orientation[-1][-1]


    def store_trajectory(self, env, actions, log_probs, done):
        
        # Prepare states
        self.take_states(env)
        self.rewards[0, 0] = env.reward_total_list[-1]

        self.done[0, 0] = done

        # Store Trajectory (Batch)
        self.states_steps = np.vstack( (self.states_steps, self.states) )                
        self.actions_steps = np.vstack( (self.actions_steps, actions) )
        self.log_probs_steps = np.vstack( (self.log_probs_steps, log_probs) )
        self.rewards_steps = np.vstack( (self.rewards_steps, self.rewards) )
        self.next_states_steps = np.vstack( (self.next_states_steps, self.next_states) )

        self.dones_steps = np.vstack( (self.dones_steps, self.done) )

        # Update State
        self.states = copy.deepcopy( self.next_states )

        # Remove the first row (init. array with zeros)
        if self.clear_flag :
            self.states_steps = np.delete( self.states_steps, 0, axis=0)            
            self.actions_steps = np.delete( self.actions_steps, 0, axis=0)
            self.log_probs_steps = np.delete( self.log_probs_steps, 0, axis=0)
            self.rewards_steps = np.delete( self.rewards_steps, 0, axis=0)
            self.next_states_steps = np.delete( self.next_states_steps, 0, axis=0)

            self.clear_flag = False

    
    def reset(self, env):
        self.clear()

        env.get_output_step(normalize_states = True)
        self.take_states(env)

    def clear(self):
        self.states = np.zeros((1, self.state_dim))        
        self.rewards = np.zeros((1, 1))
        self.next_states = np.zeros((1, self.state_dim))

        self.states_steps = np.zeros((1, self.state_dim))        
        self.actions_steps = np.zeros((1, 1))
        self.log_probs_steps = np.zeros((1, 1))
        self.rewards_steps = np.zeros((1, 1))
        self.next_states_steps = np.zeros((1, self.state_dim)) 

        self.done = np.zeros((1, 1))
        self.dones_steps = np.zeros((1, 1))

        self.clear_flag = True
        

    def get_batch(self):
        return (
            self.states_steps,
            self.actions_steps,
            self.log_probs_steps,
            self.rewards_steps,
            self.next_states_steps,
            self.dones_steps
        )
    


class ExperienceBuffer:
    def __init__(self, buffer_size=10000, success_threshold=0.7):
        self.buffer = deque(maxlen=buffer_size)
        self.success_threshold = success_threshold
        
    def add_episode(self, states, actions, rewards, dones, log_probs):
        # Calculate episode return to determine if it's a "good" episode
        episode_return = sum(rewards)
        max_return = max(rewards)  # Or some other metric you define
        
        # Only store successful episodes
        if max_return > self.success_threshold:
            episode_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'log_probs': log_probs,
                'episode_return': episode_return
            }
            self.buffer.append(episode_data)
    
    def sample_batch(self, batch_size):
        if not self.buffer:
            return None
        
        # Sample episodes and concatenate their data
        sampled_episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Concatenate all episodes' data
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        
        for episode in sampled_episodes:
            states.extend(episode['states'])
            actions.extend(episode['actions'])
            rewards.extend(episode['rewards'])
            dones.extend(episode['dones'])
            log_probs.extend(episode['log_probs'])
        
        return states, actions, rewards, dones, log_probs