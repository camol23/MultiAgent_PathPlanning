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
from aux_libs import store_model
from DRL.networks import networks_ppo_v1

sys.path.insert(0, '/home/camilo/Documents/repos/MultiAgent_application/Guidance_controller/0_single_agent_v1')


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Shared features extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.Tanh(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.Tanh(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.Tanh(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy network (Actor) - outputs action probabilities
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),  # Output
            nn.Softmax(dim=-1)
        )
        
        # Value network (Critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)            # Output
        )
        
    
    def forward(self, state):
        features = self.features(state)
        
        # Actor: Get action probabilities
        action_probs = self.policy(features)
        
        # Critic: Get state value
        value = self.value(features)
        
        return action_probs, value



class PPO_model:
    def __init__(self, state_dim, action_dim, lr=3e-4, hidden_dim=64, splitNets_flag=False):

        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=lr)

        self.actor_model = networks_ppo_v1.ActorNetwork_1(state_dim, action_dim, hidden_dim)
        self.critic_model = networks_ppo_v1.CriticNetwork_1(state_dim, hidden_dim)

        self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=lr)
        self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=0.5*lr)

        # Sheduler
        self.lr_rate = lr
        self.sheduler_flag = False
        self.scheduler_actor = None
        self.scheduler_critic = None

        # general
        self.stop_condition_flag = 0
        self.best_return = -math.inf
        self.folder_path = ""
        self.checkpoint_counter = 0
        self.splitNets_flag = splitNets_flag        # Use the separate etworks for Policy and Value Fnct
    

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


    def get_action(self, state):

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

        # print("Action = ", action)
        # print("log_prob = ", log_prob)
        
        return action.item(), log_prob.detach()


    
    def return_from_rewards(self, rewards, done, gamma, normalize_flag=True):
        '''
            Compute returns (discounted rewards)

        '''
        
        returns = []
        discounted_reward = 0

        # Extend Bool to and array
        dones = self.extend_done(rewards.shape[0], done)

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
        

    def train(self, states, actions, rewards, done, log_probs, train_config):
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
        returns = self.return_from_rewards(rewards, done, gamma, normalize_flag_returns)
        returns = torch.tensor(returns, dtype=torch.float)

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
                

                # Get current action mean, std and value
                action_probs, values = self.actor_critic(batch_states)
                
                 # Create distribution with current parameters
                dist = Categorical(action_probs)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute advantage
                advantage = batch_returns - values.squeeze()
                
                # Compute PPO policy loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                # value_loss = nn.MSELoss(values, batch_returns)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Complete loss
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), clip_grad_val)
                self.optimizer.step()


                # Scheduler
                if self.sheduler_flag :
                    # self.scheduler_actor.step()
                    # self.scheduler_critic.step()
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


    def train_splitNets(self, states, actions, rewards, done, log_probs, train_config):
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
        # buffer_flag = train_config['buffer_flag']                       # Activate the Experience Replay Buffer

        # Save Episode Total Reward
        total_iter_rewars = sum(rewards)
        self.reward_records.append(total_iter_rewars)                   # by Episode
        
        # Save Checkpoint
        if total_iter_rewars > self.best_return :
            self.save_checkpoint_splitNets()
            self.best_return = total_iter_rewars

        # Convert to tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        
        # Compute returns 
        returns = self.return_from_rewards(rewards, done, gamma, normalize_flag_returns)
        returns = torch.tensor(returns, dtype=torch.float)

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
                

                # Get current action mean, std and value
                action_probs = self.actor_model(batch_states)
                values = self.critic_model(batch_states)
                
                 # Create distribution with current parameters
                dist = Categorical(action_probs)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute advantage
                advantage = batch_returns - values.squeeze().detach()                

                # Compute PPO policy loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                # value_loss = nn.MSELoss(values, batch_returns)
                value_loss = F.mse_loss(values, batch_returns)
                
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

                # # Record Iterations
                # pi_loss_mean = policy_loss.detach().numpy()
                # val_loss_mean = value_loss.detach().numpy()
                # advantage_mean = advantage.detach().numpy().mean()
                
                # self.pi_loss_record.append( pi_loss_mean )
                # self.val_loss_record.append( val_loss_mean )
                # self.advantage_record.append( advantage_mean )                


        # Record Iterations
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

        # x_axis = np.linspace(0, len(self.reward_records)-1 , len(self.reward_records))
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

        plt.title(title)
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