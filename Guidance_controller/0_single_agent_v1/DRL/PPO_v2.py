import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F

import sys
import os
import math
from aux_libs import learning_scheduler
from aux_libs import buffers
from aux_libs import store_model
from DRL.networks import networks_ppo_v1

sys.path.insert(0, '/home/camilo/Documents/repos/MultiAgent_application/Guidance_controller/0_single_agent_v1')


'''

    Notes:
        1) To define a new model go to 'networks_ppo_v1'

'''

class PPO_model:
    def __init__(self):

        self.actor_critic = None
        self.optimizer = None

        self.actor_model = None
        self.critic_model = None

        self.opt_actor = None
        self.opt_critic = None

        # Sheduler
        self.sheduler_flag = False
        self.scheduler_actor = None
        self.scheduler_critic = None

        # Trajectory data
        self.mem_trajectory = buffers.trajectory_mem()      # Store all the data in the trajectory

        # general
        self.stop_condition_flag = 0
        self.best_return = -math.inf
        self.folder_path = ""
        self.checkpoint_counter = 0
        self.splitNets_flag = None                            # It's trained a network for the actor and another for the critic
    

        # record lists
        self.global_steps_T = 0
        self.TD_target_record = None
        self.reward_records = []
        self.pi_loss_record = []
        self.val_loss_record = []
        self.advantage_record = []

        self.lr_actorcritic_record = []
        self.lr_actor_record = []
        self.lr_critic_record = []      


    def initialize(self, model_architecture):
        '''
            main task:
                1) Load the Network
                2) Define Optimizer
                3) Set general Params
                4) Trajectory buffer

                Args:
                    1) model_architecture : Dictionary

        '''       

        state_dim = model_architecture['state_dim']
        action_dim = model_architecture['action_dim']
        hidden_dim = model_architecture['hidden_dim']
        
        lr = model_architecture['lr_rate'] 
        lr_critic = model_architecture['lr_critic']
        critc_coef = model_architecture['critic_coef_lr']        
        opti_type = model_architecture['opti_type']

        # Load the Network from 'networks_ppo_v1
        model_id = model_architecture['model_id']                             # Model to be loaded        

        # Optimizer Configuration (Details not mandatory)
        opt_settings = {
            'num_episodes' : model_architecture['num_episodes'], 
            'lr_scheduler_type' : model_architecture['lr_scheduler_type'],
            'warmup_epochs' : model_architecture['warmup_epochs']             
        }                 
            
        self.load_network(model_id, state_dim, action_dim, hidden_dim)
        self.load_optimizer(model_id, lr, lr_critic, critc_coef, opti_type, opt_settings)


        # Initialize Trajectory buffer
        self.mem_trajectory.initialize(state_dim, model_architecture['active_states'])        
    
        

        
    def load_network(self, model_id=1, state_dim=3, action_dim=3, hidden_dim=64):
        '''
            Load Models from 'networks_ppo_v1'

            **) It's mandatory have updated:
                    1) MODELS_PPO_LIST
                    2) NETWORK_CLASS_LIST
        '''

        self.splitNets_flag = networks_ppo_v1.MODELS_PPO_LIST[model_id]                                 # Check if the model is defined as two or one net
        network = networks_ppo_v1.NETWORK_CLASS_LIST[model_id]

        if self.splitNets_flag :
            self.actor_model = network[0](state_dim, action_dim, hidden_dim)
            self.critic_model = network[1](state_dim, hidden_dim)
           
        else:
            self.actor_critic = network(state_dim, action_dim, hidden_dim)            

        if self.splitNets_flag :
            net_type_str = 'split Net'
        else:
            net_type_str = 'One Net'

        print("It's loaded the Model {} type: {} ".format( model_id, net_type_str) )            



    def load_optimizer(self, model_id=1, lr=1e-3, lr_critic=0, critc_coef=0.5, opti_type='adam', opt_settings=None):
        '''
            Define Optimizer and Scheduler

                Args:
                    1) opt_settings : It's a Dictionary with the details to set the Scheduler

            **) It's mandatory have updated in 'networks_ppo_v1':
                    1) MODELS_PPO_LIST
                    2) NETWORK_CLASS_LIST
        '''

        self.splitNets_flag = networks_ppo_v1.MODELS_PPO_LIST[model_id]                                 # Check if the model is defined as two or one net
        
        if lr_critic != 0 :
            critc_coef = 1
        else:
            lr_critic = lr


        if opti_type == 'adamw' :
            
            if self.splitNets_flag :
                self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=lr)
                self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=critc_coef*lr_critic)
            else:
                self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=lr)
            
        elif opti_type == 'adam' :

            if self.splitNets_flag :
                self.opt_actor = torch.optim.Adam(self.actor_model.parameters(), lr=lr)
                self.opt_critic = torch.optim.Adam(self.critic_model.parameters(), lr=critc_coef*lr_critic)
            else:
                self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        
        # Define Scheduler
        if opt_settings == None :
            lr_scheduler_type = 'cte'
        else:
            lr_scheduler_type = opt_settings['lr_scheduler_type']
            num_episodes = opt_settings['num_episodes']
            

        ## Cosine Scheduler
        if lr_scheduler_type == 'cosine' :
            self.sheduler_flag = True

            warmup_epochs = opt_settings['warmup_epochs']

            if self.splitNets_flag :
                self.scheduler_actor = learning_scheduler.CosineWarmupScheduler(self.opt_actor, warmup_epochs, num_episodes)
                self.scheduler_critic = learning_scheduler.CosineWarmupScheduler(self.opt_critic, warmup_epochs, num_episodes)
            else:
                self.scheduler_actor = learning_scheduler.CosineWarmupScheduler(self.optimizer, warmup_epochs, num_episodes)


        print("Optimizer loaded = {} with Scheduler = {}".format( opti_type, lr_scheduler_type ))



    def get_action(self, state, training=True):

        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.splitNets_flag :
                action_probs = self.actor_model(state)
            else:
                action_probs, _ = self.actor_critic(state)
                  
            # Create categorical distribution
            dist = Categorical(action_probs)
            
            # Sample single action
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Just for Test 
            print("action_probs ", action_probs)
            action = np.argmax(action_probs)
            action_probs_sq = action_probs.squeeze()
            print('action_probs_sq', action_probs_sq, action.item(), action_probs_sq[int(action.item()) ])
            log_prob = torch.log(action_probs_sq[int(action.item()) ])
            # print("action ", action)
            # if not(training) :
            #     action = np.argmax(action_probs)
                # print("action_probs ", action_probs)



        print("Action = ", action)
        print("log_prob = ", log_prob)
        
        return action.item(), log_prob.detach()


    
    def return_from_rewards(self, rewards, dones, gamma, normalize_flag=True):
        '''
            Compute returns (discounted rewards)

        '''
        
        returns = []
        discounted_reward = 0        

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done :
                discounted_reward = 0

            discounted_reward = reward + gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        # returns = torch.FloatTensor(returns)
        returns = np.array(returns)
        
        # Normalize returns
        if normalize_flag :
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def compute_vals_gae(self, states, next_states):
        '''
            Compute values for GAE (consider SplitNet_Flag)

                1) No grads
                2) squeeze()

            
                Returns
                    1) values
                    2) next_values
        '''
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)

        with torch.no_grad():

            if self.splitNets_flag :
                values = self.critic_model(states)
                next_values = self.critic_model(next_states)
            else:
                _, values = self.actor_critic(states)
                _, next_values = self.actor_critic(next_states)

            values.squeeze().detach()
            next_values.squeeze().detach()

        return values, next_values


    def compute_gae(self, rewards, states, next_states, dones, gamma, gae_lambda, norm_flag=True):
        """
        Compute Generalized Advantage Estimation (GAE)

        Note:
            Update 'comput_vals_gae()' function

        Returns:
            advantages: GAE for each timestep
            returns: Returns for each timestep
        """
        advantages = []
        gae = 0

        # print('next_State', next_states.shape)
        # print('STATE', states.shape)
        values, next_values = self.compute_vals_gae(states, next_states)
        
        # print('next_values', next_values.shape)
        
        # For each timestep in reversed order
        for t in reversed(range(len(rewards))):
            # If it's the last timestep, next_value is the value of the next state
            # Otherwise, it's the value of the current state
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # Calculate TD error for timestep t
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # Calculate GAE for timestep t
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)                                                                                  # Insert this advantage at the beginning of the list
            

        # advantages = torch.FloatTensor(advantages)        
        # returns = advantages + torch.FloatTensor(values)                                                               # Calculate returns as advantages + values
        returns = advantages + values

        # Normalize advantages
        if norm_flag :
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


    def select_return(self, rewards, states, next_states, dones, gamma, gae_lambda, return_type='return', norm_returns=True):            
        

        if return_type == 'gae' :
            advantages, returns = self.compute_gae(rewards, states, next_states, dones, gamma, gae_lambda, norm_returns)
        else:
            # 'return'
            returns = self.return_from_rewards(rewards, dones, gamma, norm_returns)
            # Not apply 
            advantages = np.zeros_like(returns)

        
        return returns, advantages


    def train(self, train_config):
        '''
            Compute Losses, gradients, and update weigths

            Note:
                1) Mini-Batch performed

        '''

        # Training Parameters and Configuration Settings
        gamma = train_config['gamma']
        epsilon = train_config['epsilon']                               # PPO clipping parameter
        epochs = train_config['epochs']                                 # Number of epochs per update
        batch_size = train_config['batch_size']                         # Mini-Batch size
        normalize_flag_returns = train_config['norm_returns']           # Normalize computed returns
        entropy_coef = train_config['entropy_coef']                     # Entropy Coeffitient for Pi_loss
        clip_grad_val = train_config['clip_grad_val']                   # Clipping value for Gradients
        val_loss_coef = train_config['val_loss_coef']                   # Factor to multiply val. loss        
        return_type = train_config['return_type']                       # Slect between functions 
        gae_lambda = train_config['gae_lambda']                         # GAE return

        # Bring Trajectory Data
        states, actions, log_probs, rewards, next_states, done  = self.mem_trajectory.get_batch()
        
        # Extend Bool to and array
        # done = True
        # done = self.extend_done(rewards.shape[0], done)

        # Save Episode Total Reward
        total_iter_rewars = sum(rewards)
        self.reward_records.append(total_iter_rewars)                   # by Episode
        
        # Save Checkpoint
        if total_iter_rewars > self.best_return :
            self.save_checkpoint()
            self.best_return = total_iter_rewars

        # Convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        
        # Compute returns 
        returns, advantages = self.select_return(rewards, states, next_states, done, gamma, gae_lambda, return_type, normalize_flag_returns)
        returns = torch.tensor(returns, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)

        # PPO update for specified number of epochs
        for _ in range(epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                

                # Get current action mean, std and value
                action_probs, values = self.actor_critic(batch_states)
                
                 # Create distribution with current parameters
                dist = Categorical(action_probs)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute advantage
                if return_type == 'gae' :
                    advantage = batch_advantages
                else:
                    advantage = batch_returns - values.squeeze().detach()
                
                # Compute PPO policy loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                # value_loss = nn.MSELoss(values, batch_returns)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Complete loss
                loss = policy_loss + val_loss_coef * value_loss - entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), clip_grad_val)
                self.optimizer.step()


            # Scheduler
            if self.sheduler_flag :
                self.scheduler_actor.step()
                self.scheduler_critic.step()
                lr_item = self.scheduler_actor.optimizer.param_groups[0]['lr']
            else:
                lr_item = self.optimizer.param_groups[0]['lr']
            
            self.lr_actorcritic_record.append(lr_item)


            # Record Iterations
            pi_loss_mean = policy_loss.detach().numpy()
            val_loss_mean = value_loss.detach().numpy()
            advantage_mean = advantage.detach().numpy().mean()
            
            self.pi_loss_record.append( pi_loss_mean )
            self.val_loss_record.append( val_loss_mean )
            self.advantage_record.append( advantage_mean )                


        
        # Visualization by Episode
        print()
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        print("     Pi Loss = {}  Val. Loss = {} ".format(pi_loss_mean, val_loss_mean))
        print("___________________________________")
        self.global_steps_T += 1                


    def train_splitNets(self, train_config):
        '''
            Compute Losses, gradients, and update weigths for :
                    Actor and Critic idependent networks

            Note:
                1) Mini-Batch performed

        '''

        # Training Parameters and Configuration Settings
        gamma = train_config['gamma']
        epsilon = train_config['epsilon']                               # PPO clipping parameter
        epochs = train_config['epochs']                                 # Number of epochs per update
        batch_size = train_config['batch_size']                         # Mini-Batch size
        normalize_flag_returns = train_config['norm_returns']           # Normalize computed returns
        entropy_coef = train_config['entropy_coef']                     # Entropy Coeffitient for Pi_loss
        clip_grad_val = train_config['clip_grad_val']                   # Clipping value for Gradients        
        val_loss_coef = train_config['val_loss_coef']                   # Factor to multiply val. loss        
        return_type = train_config['return_type']                       # Slect between functions 
        # buffer_flag = train_config['buffer_flag']                       # Activate the Experience Replay Buffer

        # Bring Trajectory Data
        states, actions, log_probs, rewards, next_states, done  = self.mem_trajectory.get_batch()


        # Save Episode Total Reward
        total_iter_rewars = sum(rewards)
        self.reward_records.append(total_iter_rewars)                   # by Episode

        # Extend Bool to and array
        # done = True
        # done = self.extend_done(rewards.shape[0], done)
        
        # Save Checkpoint
        if total_iter_rewars > self.best_return :
            self.save_checkpoint_splitNets()
            self.best_return = total_iter_rewars

        # Convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        
        # Compute returns 
        returns, advantages = self.select_return(rewards, states, next_states, done, gamma, return_type, normalize_flag_returns)
        returns = torch.tensor(returns, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)

        # PPO update for specified number of epochs
        for _ in range(epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                idx = indices[start_idx:start_idx + batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                

                # Get current action mean, std and value
                action_probs = self.actor_model(batch_states)
                values = self.critic_model(batch_states)
                
                 # Create distribution with current parameters
                dist = Categorical(action_probs)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute advantage
                if return_type == 'gae' :
                    advantage = batch_advantages
                else:
                    advantage = batch_returns - values.squeeze().detach()                

                # Compute PPO policy loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                # value_loss = nn.MSELoss(values, batch_returns)
                value_loss = val_loss_coef * F.mse_loss(values, batch_returns)
                
                # Complete loss
                policy_loss = policy_loss - entropy_coef * entropy
                
                # Update network
                self.opt_actor.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_model.parameters(), clip_grad_val)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_model.parameters(), clip_grad_val)
                self.opt_critic.step()

                # for name, param in self.actor_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} gradient norm: {param.grad.norm().item()}")
                #     else:
                #         print(f"{name} has no gradient!")


                # # Record Iterations
                # pi_loss_mean = policy_loss.detach().numpy()
                # val_loss_mean = value_loss.detach().numpy()
                # advantage_mean = advantage.detach().numpy().mean()
                
                # self.pi_loss_record.append( pi_loss_mean )
                # self.val_loss_record.append( val_loss_mean )
                # self.advantage_record.append( advantage_mean )                


        # Record Iterations
        if done[-1, -1] :
            pi_loss_mean = policy_loss.detach().numpy()
            val_loss_mean = value_loss.detach().numpy()
            advantage_mean = advantage.detach().numpy().mean()
            
            self.pi_loss_record.append( pi_loss_mean )
            self.val_loss_record.append( val_loss_mean )
            self.advantage_record.append( advantage_mean )           

            # Scheduler
            if self.sheduler_flag :
                self.scheduler_actor.step()
                self.scheduler_critic.step()
                lr_actor_item = self.scheduler_actor.optimizer.param_groups[0]['lr']
                lr_critic_item = self.scheduler_critic.optimizer.param_groups[0]['lr']
            else:
                lr_actor_item = self.opt_actor.param_groups[0]['lr']
                lr_critic_item = self.opt_critic.param_groups[0]['lr']
            
            self.lr_actorcritic_record.append(lr_actor_item)
            self.lr_critic_record.append(lr_critic_item)

            # Visualization by Episode
            print()
            print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
            print("     Pi Loss = {}  Val. Loss = {} ".format(pi_loss_mean, val_loss_mean))
            print(" Entropy = {}".format(entropy))
            print()
            # print("___________________________________")
            self.global_steps_T += 1                



    
    def extend_done(self, size, done):
        '''
            Extend Val. as the action vector
                1) The Val is just assigned to the last element
                    in the numpy array
                2) All elements are zero
        '''

        done_extended = np.zeros((size, 1))
        done_extended[-1, -1] = done

        return done_extended
    
    def save_checkpoint(self):

        # Count the Total of Folders
        checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1'    
        if self.checkpoint_counter == 0 :

            checkpoint_folders = [name for name in os.listdir(checkpoint_path) ]
            num_folders = len(checkpoint_folders)
            num_folders = num_folders + 1 

            # Create the folder for the current test
            folder_name = '/model_ppo_v1_test_' + str(num_folders)
            self.folder_path = checkpoint_path+folder_name
            os.makedirs(self.folder_path)
        
        # Independent Folders
        actorCritic_path = self.folder_path + '/actorCritic_v1'

        # Name equ = file_name + str(episode) + ".pt"
        file_name_actor = "checkpoint_episode_" + str(self.global_steps_T) + "_reward_" 
        
        store_model.save_model(self.actor_critic, self.optimizer, self.reward_records[-1], self.reward_records, actorCritic_path, file_name_actor)        
        

        self.checkpoint_counter += 1 
        print("Saved as = ", actorCritic_path + "/" + file_name_actor)




    def plot_training(self, episodes = "", steps = "", title=""):
        '''
            Plot in a row:
                (1) Reward
                (2) Pi. Loss
                (3) Val. Loss
        '''
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 4))

        ave_reward = ave_array(self.reward_records, window_size=4)
        axes[0, 0].plot(self.reward_records, 'r',  alpha=0.2)
        axes[0, 0].plot(ave_reward, 'r')
        axes[0, 0].set_title("Sum. rewards by Episode " + str(episodes) + " - Steps " + str(steps) )

        axes[0, 1].plot(self.pi_loss_record, 'g')
        axes[0, 1].set_title("Pi. loss")

        axes[0, 2].plot(self.val_loss_record, 'b')
        axes[0, 2].set_title("Val. loss")

        # Second Row
        axes[1, 0].plot(self.advantage_record, 'b')
        axes[1, 0].set_title("Advanage mean")

        axes[1, 1].plot(self.lr_actorcritic_record, 'r')
        axes[1, 1].set_title("Lr. Actor")

        axes[1, 2].plot(self.lr_critic_record, 'r')
        axes[1, 2].set_title("Lr. Critic")
        

        # Adjust layout
        for axe in axes:
            for ax in axe :
                # ax.set_aspect('equal')
                ax.grid(True)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()



    def save_checkpoint_splitNets(self):

        # Count the Total of Folders
        checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1_splitNets'    
        if self.checkpoint_counter == 0 :

            checkpoint_folders = [name for name in os.listdir(checkpoint_path) ]
            num_folders = len(checkpoint_folders)
            num_folders = num_folders + 1 

            # Create the folder for the current test
            folder_name = '/model_ppo_v1_test_' + str(num_folders)
            self.folder_path = checkpoint_path+folder_name
            os.makedirs(self.folder_path)
        
        # Independent Folders
        actor_path = self.folder_path + '/actor_ppo_v1'
        critic_path = self.folder_path +  '/critic_ppo_v1'

        # Name equ = file_name + str(episode) + ".pt"
        file_name_actor = "checkpoint_episode_" + str(self.global_steps_T) + "_reward_" 
        file_name_critic = "checkpoint_episode_" + str(self.global_steps_T) + "_reward_" 
        
        store_model.save_model(self.actor_model, self.opt_actor, self.reward_records[-1], self.reward_records, actor_path, file_name_actor)
        store_model.save_model(self.critic_model, self.opt_critic, self.reward_records[-1], self.reward_records[-1], critic_path, file_name_critic)
        

        self.checkpoint_counter += 1 
        print("Saved as = ", actor_path + "/" + file_name_actor)



def ave_array(reward_list, window_size=4):
    '''
        Creates a list with the average valuse with the same size the input

    '''

    average = [] 
    
    for i in range(0, len(reward_list), window_size) :

        lapse = reward_list[i : i+window_size]
        mean_val = np.mean(lapse)

        for _ in range(0, len(lapse)) :
            average.append(mean_val.item())


    return average