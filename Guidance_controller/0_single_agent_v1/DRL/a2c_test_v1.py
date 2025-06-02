import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

'''
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
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits


# Value V(s) function
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value



class drl_model:
    def __init__(self):

        self.actor_model = ActorNet().to(device)
        self.critic_model = ValueNet().to(device)

        self.load_optimizers()

        # general
        self.stop_condition_flag = 0

        # record lists
        self.reward_records = []
        self.global_steps_T = 0



    # pick up action with above distribution policy_pi
    def pick_sample(self, s_batch):
        with torch.no_grad():
            #   --> size : (1, 4)
            # s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            # print(s_batch)
            # Get logits from state
            #   --> size : (1, 2)
            logits = self.actor_model(s_batch)
            # print("Porbs = ", logits)
            #   --> size : (2)
            logits = logits.squeeze(dim=0)
            # From logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Pick up action's sample
            a = torch.multinomial(probs, num_samples=1)
            # Return
            # return a.tolist()[0]
            return a
        


    #### Training 
    def load_optimizers(self):
        self.opt_critic = torch.optim.AdamW(self.critic_model.parameters(), lr=0.001)
        self.opt_actor = torch.optim.AdamW(self.actor_model.parameters(), lr=0.001)


    def training_a2c(self, states_list, actions_list, rewards_list):
        '''
            Compute Losses, gradients, and update weigths

            Note:
                1) One Iteration
        '''
        gamma = 0.99                                    # Discount factor


        #
        # Get cumulative rewards (Return)
        #
        cum_rewards = np.zeros_like(rewards_list)
        reward_len = len(rewards_list)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards_list[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

        #
        # Train (optimize parameters)
        #

        # Optimize value loss (Critic)
        self.opt_critic.zero_grad()

        states = torch.tensor(states_list, dtype=torch.float).to(device)
        cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)

        values = self.critic_model(states)
        values = values.squeeze(dim=1)
        vf_loss = F.mse_loss(
            values,
            cum_rewards,
            reduction="none")
        
        vf_loss.sum().backward()
        self.opt_critic.step()


        # Optimize policy loss (Actor)
        with torch.no_grad():
            values = self.critic_model(states)
        self.opt_actor.zero_grad()

        actions = torch.tensor(actions_list, dtype=torch.int64).to(device)
        advantages = cum_rewards - values
        
        logits = self.actor_model(states)
        logits = torch.squeeze(logits)
        actions = torch.squeeze(actions)
        print(actions.shape)
        print(logits.shape)
        log_probs = -F.cross_entropy(logits, actions, reduction="none")
        pi_loss = -log_probs * advantages
        
        pi_loss.sum().backward()
        self.opt_actor.step()

        # Output total rewards in episode (max 500)
        # print("Run episode{} with rewards {}".format(i, sum(rewards_list)), end="\r")

        total_iter_rewars = sum(rewards_list)
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        
        self.reward_records.append(total_iter_rewars) # by epoch
        self.global_steps_T += 1

        # print(self.global_steps_T)

        # stop if reward mean > 475.0
        # self.stop_condition()

    
    def stop_condition(self):
        # if np.average(self.reward_records[-50:]) > 475.0:
        #     self.stop_condition_flag = 1
        pass
        

    def preprossed_states():
        pass


    def plot_results(self):
        
        plt.plot(self.reward_records)
        plt.title("Sum. rewards by epoch - Epochs " + str(self.global_steps_T) )
        plt.show() 
