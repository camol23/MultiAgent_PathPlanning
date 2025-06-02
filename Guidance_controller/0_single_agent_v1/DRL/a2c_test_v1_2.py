import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from aux_libs import learning_scheduler
from aux_libs import store_model

sys.path.insert(0, '/home/camilo/Documents/repos/MultiAgent_application/Guidance_controller/0_single_agent_v1')

'''
    Training Algorithm follows the fundamental 
    Asynchronous Advantage Actor-Critic from : 
            *) Asynchronous Methods for Deep Reinforcement Learning:
               https://arxiv.org/pdf/1602.01783

    * V_1.2 : It's made to apply some possible corrections
                over V_1.0  

    * Main Style : Batch are formed by the steps samples

        Action Space: 
                        1) Move to the ritgh (5 degrees)
                        2) Move to the left (5 degrees) 

        Observation Space:
                        1) Heading Error (angle) 
                        2) Distance to the goal 

        Reward Space:
                        1) Inverse proportional to the Error
                        2) Inverse proportional to the Distance

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Policy function
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

        self.soft_layer = nn.Softmax(dim=-1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden_2(outs)
        outs = F.relu(outs)
        outs = self.hidden_3(outs)
        outs = F.relu(outs)

        logits = self.output(outs)
        probs = self.soft_layer(logits)

        return probs


# Value V(s) function
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden_2(outs)
        outs = F.relu(outs)
        outs = self.hidden_3(outs)
        outs = F.relu(outs)
        value = self.output(outs)
        return value


class drl_model:
    def __init__(self):

        self.actor_model = ActorNet().to(device)
        self.critic_model = ValueNet().to(device)

        self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=0.001)
        self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=0.001)

        # Sheduler
        self.lr_rate = 0.001
        self.sheduler_flag = False
        self.scheduler_actor = None
        self.scheduler_critic = None

        # general
        self.stop_condition_flag = 0
        self.best_return = 0
        self.folder_path = ""
        self.checkpoint_counter = 0
        self.last_checkpoint_name_actor = ""                # File name to recover the last best model ("Buffer replay")
        self.last_checkpoint_name_critic = ""               # File name to recover the last best model 
        self.counter_best = 0                               # Limint of no Model Improvements ()
        self.recover_episode = []                           # The episode where the Model Recovery is done
    

        # record lists
        self.reward_records = []
        self.global_steps_T = 0
        self.TD_target_record = None
        self.pi_loss_record = []
        self.val_loss_record = []
        self.advantage_record = []

        self.lr_actor_record = []
        self.lr_critic_record = []


    def load_newModel(self, actor, critic, lr_sheduler_flag=False, warmup_epochs=5, total_epochs=30, lr_rate=0):
        
        self.actor_model = actor.to(device)
        self.critic_model = critic.to(device)

        # Optimizer
        if lr_rate == 0 :
            lr_rate = self.lr_rate
        
        self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=lr_rate)
        self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=0.5*lr_rate)
        
        if lr_sheduler_flag :
            self.sheduler_flag = True

            self.scheduler_actor = learning_scheduler.CosineWarmupScheduler(self.opt_actor, warmup_epochs, total_epochs)
            self.scheduler_critic = learning_scheduler.CosineWarmupScheduler(self.opt_critic, warmup_epochs, total_epochs)


        print("Model Updated ...")
    

    def pick_sample(self, s_batch, vis_flag = False, vis_vals=False):
        '''
            Pick a Sample from action-space with the Policy_pi Probs.

                1) Actor-Net Output : Softmax
                2) Method:            np.random.choice

        '''
        self.actor_model.eval()  # This changes how BatchNorm layers behave
        with torch.no_grad():

            # s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)           #   --> size : (1, 2)
            
            # Apply Policy - Get Probs for the action Space
            probs = self.actor_model(s_batch)                                #   --> size : (1, 2/3)
            
            #   --> size : (2)
            action_probs = probs.squeeze()            
            action_probs = action_probs.numpy()
            
            # Pick an action based on the Probs
            action = np.random.choice(len(action_probs), p=action_probs)


            if vis_flag :
                
                print()
                if vis_vals :
                    print("--------- Pick-Sample Function ---------")
                    print("Input States = ", s_batch.shape, s_batch)
                    print("Actor Output = ", probs.shape, probs)
                    print("Squeeze OP.  = ", action_probs.shape, action_probs)
                    print("Action       = ", action)
                else:
                    print("--------- Pick-Sample Function ---------")
                    print("Input States = ", s_batch.shape)
                    print("Actor Output = ", probs.shape)
                    print("Squeeze OP.  = ", action_probs.shape)
                    print("Action       = ", action)

                print("-----------------------------------------")
                print()
                
            
            return action
        


    #### Training 
    def load_optimizers(self, opt_actor, opt_critic):
        self.opt_critic = opt_critic
        self.opt_actor = opt_actor


    def TD_target_1(self, rewards_list, gamma, reverse_flag=False, norm_flag= False):
        '''
            
            reverse:
                a) True: cum_rewards goes from r_[t_last] to r_[0] 
                b) True: cum_rewards goes from r_[0] to r_[t_last] 
        '''
        
        cum_rewards = np.zeros_like(rewards_list)
        reward_len = len(rewards_list)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards_list[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

        # Not necessary (Just for test)
        if reverse_flag :   
            cum_rewards = np.flip(cum_rewards) 

        # Reward Clipping/Normalization
        # Looks like enhance the response
        if norm_flag :
            cum_rewards = (cum_rewards - cum_rewards.mean()) / (cum_rewards.std() + 1e-8)

        return cum_rewards
    

    def compute_n_step_returns(self, rewards, values, gamma=0.99, n_steps=5):
        
        returns = []
        len_rewards = len(rewards)
        values = values.squeeze().numpy()

        if len_rewards <= n_steps :
            n_steps = len_rewards - 1

        for t in range(len_rewards):
            # Calculate n-step return
            R = 0
            for i in range(n_steps):
                if t + i < len_rewards :
                    R += (gamma ** i) * rewards[t + i]

            # Add bootstrap value if available
            if t + n_steps < len(values):
                R += (gamma ** n_steps) * values[t + n_steps]
            
            returns.append(R)
        
        return returns


    def TD_target_2(self, states, rewards_list, gamma):
        '''
            Use Critic network to compute V( S_[t+1] )
        '''
        
        values = self.critic_model(states).detach().numpy()

        # G = r_[t] + gamma * V(S_[t+1])
        td_target = np.zeros_like(rewards_list)
        td_target[0:-1] = rewards_list[0:-1] + gamma*values[1:] 

        # V(S_[last+1]) = 0
        td_target[-1] = rewards_list[-1] 

        return td_target



    def TD_target_3(self, values, rewards_list, gamma):
        '''
            Extract V( S_[t+1] ) from V(S_[t]) vector

                values: V( S_[t+1] )
        '''
        
        values_next = values.detach()
        
        # G = r_[t] + gamma * V(S_[t+1])
        td_target = np.zeros_like(rewards_list)
        td_target[0:-1] = rewards_list[0:-1] + gamma*values_next[1:] 
        
        # V(S_[last+1]) = 0
        td_target[-1] = rewards_list[-1] 

        return td_target


    def training_a2c(self, states_list, actions_list, rewards_list, vis_flag=False, clip_grad_flag=True, back_grad_mean=True, buffer_limit=10):
        '''
            Compute Losses, gradients, and update weigths

            Note:
                1) One Iteration
        '''
        
        # Buffer Reaply - based on Checkpoints
        total_iter_rewars = sum(rewards_list)
        self.reward_records.append(total_iter_rewars) # by epoch
        
        # Save Checkpoint
        if total_iter_rewars > self.best_return :
            self.save_checkpoint(states_list, actions_list, rewards_list)
            self.best_return = total_iter_rewars
            
            self.counter_best = 0
        else:
            # Load The best Model previous Computed
            if self.counter_best > buffer_limit :
                checkpoint_dict = self.load_checkpoint()
                self.counter_best = 0

                # recovery the last experience
                states_list = checkpoint_dict['states_history']
                actions_list = checkpoint_dict['actions_history']
                rewards_list = checkpoint_dict['last_rewards']

                self.recover_episode.append( len(self.reward_records) )
                print("Recovery is done in EPisode = ", self.recover_episode)
            else:
                self.counter_best += 1


        self.actor_model.train()

        gamma = 0.99                                                                            # Discount factor
        clip_grad = 1.0  # 0.5                                                              
        states = torch.tensor(states_list, dtype=torch.float).to(device)
        # actions = torch.tensor(actions_list, dtype=torch.int64).to(device)


        with torch.no_grad():
            values_detached = self.critic_model(states)

        
        # Get cumulative rewards (Return G) in a List
        # td_target = self.TD_target_1(rewards_list, gamma, reverse_flag=False, norm_flag=False)             # Using just rewards        
        td_target = self.compute_n_step_returns(rewards_list, values_detached, gamma, 10)                    # Usign Bootstrap
        # td_target = self.TD_target_2(states, rewards_list, gamma)                                          # Using Critic network        
        td_target = torch.tensor(td_target, dtype=torch.float).to(device)
        

        # Compute Values
        self.critic_model.train()                                                               # necessary?
        values = self.critic_model(states)
        # values = values.squeeze(dim=1)                                                        # Require when I used states([batch, 1, 2])

        # Optimize value loss (Critic)
        vf_loss = F.mse_loss(
            values,
            td_target,
            reduction="none")                                               # In this case requires loss.sum() or loss.mean()
        
        #print("Loss Val.shape = ", vf_loss.shape)
        self.opt_critic.zero_grad()
        if back_grad_mean :
            vf_loss.mean().backward()
        else:
            vf_loss.sum().backward()
        
        if clip_grad_flag :
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=clip_grad)
        self.opt_critic.step()


        # Optimize policy loss (Actor)        
        advantages = (td_target - values_detached).detach()
        
        # Train Actor
        self.actor_model.train()                   # necessary?
        action_probs = self.actor_model(states)
        action_probs = torch.squeeze(action_probs)
        
        # Prepare index to target the Probs                             
        num_actions = actions_list.shape[0]
        actions = np.squeeze(actions_list)                                                      # Size(Batch + #Agents)

        # Batch size = 1
        stability_cte_log_loss = 1e-10
        if num_actions == 1 :
            # log_probs = torch.log(action_probs[actions] + stability_cte_log_loss)
            action_probs_selected = action_probs[actions]
        else:
            idx_rows = np.arange(len(actions))                                                  # [0, ... , Total_samples]
            action_probs_selected = action_probs[idx_rows, actions]
            # log_probs = torch.log(action_probs[idx_rows, actions] + stability_cte_log_loss)
            # print("idx_rows", idx_rows.shape)

        log_probs = torch.log(action_probs_selected + stability_cte_log_loss)        
         
        # Calculate entropy of the policy 
        # (It's suggested with all the probs not just the selected)
        # Suggestion :   entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        entropy_coef = 0.01
        entropy = -(action_probs_selected * log_probs)                                          # Array
        
        pi_loss = -(log_probs * advantages) - entropy_coef * entropy 
        
        self.opt_actor.zero_grad()
        if back_grad_mean :
            pi_loss.mean().backward()
        else:
            pi_loss.sum().backward()        

        if clip_grad_flag :
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=clip_grad)
        self.opt_actor.step()

        # Scheduler
        if self.sheduler_flag :
            self.scheduler_actor.step()
            self.scheduler_critic.step()

            lr_actor_item = self.scheduler_actor.optimizer.param_groups[0]['lr']
            lr_critic_item = self.scheduler_critic.optimizer.param_groups[0]['lr']
            self.lr_actor_record.append(lr_actor_item)
            self.lr_critic_record.append(lr_critic_item)

        # stop if reward mean > 475.0
        # self.stop_condition()
        
        # Record Iterations
        pi_loss_mean = torch.mean(pi_loss).detach().numpy()
        val_loss_mean = torch.mean( torch.squeeze(vf_loss) ).detach().numpy()
        advantage_mean = torch.mean(advantages).item()
        

        self.TD_target_record = td_target
        self.pi_loss_record.append( pi_loss_mean )
        self.val_loss_record.append( val_loss_mean )
        self.advantage_record.append( advantage_mean )


        # Visualization
        print()
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        print("     Pi Loss = {}  Val. Loss = {} ".format(pi_loss_mean, val_loss_mean))
        print("___________________________________")
        self.global_steps_T += 1

        if vis_flag :
            print("--------- Training Function ---------")
            print("Input States Shape  = ", states_list.shape)
            print("Input Actions Shape = ", actions_list.shape)
            print("Input Rewards Shape = ", rewards_list.shape)
            print()
            print("TD Target Shape    = ", td_target.shape)
            print("Actor Output Shape = ", action_probs.shape)
            print("Log( Probs ) Shape = ", log_probs.shape)
            

    
    def stop_condition(self):
        # if np.average(self.reward_records[-50:]) > 475.0:
        #     self.stop_condition_flag = 1
        pass
        

    def preprossed_states():
        pass


    def plot_rewards(self):
        
        plt.plot(self.reward_records)
        plt.title("Sum. rewards by epoch - Epochs " + str(self.global_steps_T) )
        plt.show() 


    def plot_training(self, episodes = "", steps = ""):
        '''
            Plot in a row:
                (1) Reward
                (2) Pi. Loss
                (3) Val. Loss
        '''
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 4))

        # First ROw
        axes[0, 0].plot(self.reward_records, 'r')
        # axes[0, 0].set_title("Sum. rewards by Episode - Epochs " + str(self.global_steps_T) + " - " + str(steps) )
        axes[0, 0].set_title("Sum. rewards by Episode " + str(episodes) + " - Steps " + str(steps) )

        axes[0, 1].plot(self.pi_loss_record, 'g')
        axes[0, 1].set_title("Pi. loss")

        axes[0, 2].plot(self.val_loss_record, 'b')
        axes[0, 2].set_title("Vsal. loss")

        # Second Row
        axes[1, 0].plot(self.advantage_record, 'b')
        axes[1, 0].set_title("Advanage mean")

        axes[1, 1].plot(self.lr_actor_record, 'r')
        axes[1, 1].set_title("Lr. Actor")

        axes[1, 2].plot(self.lr_critic_record, 'r')
        axes[1, 2].set_title("Lr. Critic")
        

        # Adjust layout
        for axe in axes:
            for ax in axe :
                # ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()


    def save_checkpoint(self, states=None, actions=None, rewards=None):

        # Count the Total of Folders
        checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_v1_2'    
        if self.checkpoint_counter == 0 :

            checkpoint_folders = [name for name in os.listdir(checkpoint_path) ]
            num_folders = len(checkpoint_folders)
            num_folders = num_folders + 1 

            # Create the folder for the current test
            folder_name = '/model_v1_2_test_' + str(num_folders)
            self.folder_path = checkpoint_path+folder_name
            os.makedirs(self.folder_path)
        
        # Independent Folders
        actor_path = self.folder_path + '/actor_v1_2'
        critic_path = self.folder_path +  '/critic_v1_2'

        # Name equ = file_name + str(episode) + ".pt"
        file_name_actor = "checkpoint_episode_" + str(self.global_steps_T) + "_reward_" 
        file_name_critic = "checkpoint_episode_" + str(self.global_steps_T) + "_reward_" 
        
        store_model.save_model(self.actor_model, self.opt_actor, self.reward_records[-1], self.reward_records, actor_path, file_name_actor, states, actions, rewards)
        store_model.save_model(self.critic_model, self.opt_critic, self.reward_records[-1], self.reward_records, critic_path, file_name_critic)
        

        self.checkpoint_counter += 1 

        self.last_checkpoint_name_actor = actor_path + "/" + file_name_actor + str(self.reward_records[-1]) + '.pt'
        self.last_checkpoint_name_critic = critic_path + "/" + file_name_critic + str(self.reward_records[-1]) + '.pt'
        # print("Saved as = ", actor_path + "/" + file_name_actor)
        print("Saved as = ", self.last_checkpoint_name_actor)



    def load_checkpoint(self):
        '''
            Recover The best shot

        '''    

        self.actor_model, _, checkpoint_dict_actor = store_model.load_model(self.actor_model, self.opt_actor, self.last_checkpoint_name_actor)
        self.critic_model, _, _ = store_model.load_model(self.critic_model, self.opt_critic, self.last_checkpoint_name_critic)

        print("It's recovery the next model = ", self.last_checkpoint_name_actor)

        return checkpoint_dict_actor
