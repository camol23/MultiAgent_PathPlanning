import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Actor-Critic Network with:
            1) Separate actor and critic heads
            2) Common Feature extration (input network)
        
        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),

            nn.Softmax(dim=-1)                              # Probability distribution over actions
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                        # Estimate state value
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): Input state
        
        Returns:
            action_probs (torch.Tensor): Probability distribution over actions
            state_value (torch.Tensor): Estimated state value
        """
        features = self.feature_extractor(state)

        # Actor-Critic
        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value
    

class ActorCritic_Agent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        """
        Actor-Critic Agent with training logic
        
        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            gamma    (float): Discount factor for future rewards
            learning_rate (float): Learning rate for optimization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network and optimizer
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma

        # General use
        self.global_steps_T = 0

        # record lists
        self.reward_records = []
        self.pi_loss_record = []
        self.val_loss_record = []
        self.loss_record = []
        self.advantage_record = []
        self.cumulative_reward_record = [0]
        self.TD_target_record = []

    
    def select_action(self, state, vis_flag = False, vis_vals=False):
        """
        Select action based on current policy
        
        Args:
            state (numpy.ndarray): Current environment state
        
        Returns:
            action (int): Selected action
            action_prob (float): Probability of selected action
        """
        
        state = torch.FloatTensor(state).to(self.device)
        # state = torch.FloatTensor(state).to(self.device)
        
        # probs, _ = self.network(state)
        self.network.eval()  # This changes how BatchNorm layers behave
        with torch.no_grad():
            probs, _ = self.network(state)
        
        action_probs = torch.squeeze(probs).detach().cpu().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)


        if vis_flag :    
            print()
            if vis_vals :
                print("--------- Pick-Sample Function ---------")
                print("Input States = ", state.shape, state)
                print("Actor Output = ", probs.shape, probs)
                print("Squeeze OP.  = ", action_probs.shape, action_probs)
                print("Action       = ", action)
            else:
                print("--------- Pick-Sample Function ---------")
                print("Input States = ", state.shape)
                print("Actor Output = ", probs.shape)
                print("Squeeze OP.  = ", action_probs.shape)
                print("Action       = ", action)

            print("-----------------------------------------")
            print()

        return action, action_probs[action]
    
    def train_batch(self, state, action, reward, next_state, done, vis_flag=False):
        """
        Perform a training for a Batch using Actor-Critic algorithm
        
        Args:
            state   (numpy): Current state
            action    (numpy): Action taken
            reward  (numpy): Reward received
            next_state (numpy): Next state
            done (bool): Episode termination flag
        """
        self.network.train()

        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # action = torch.tensor(action).to(self.device)
        done_flag = done

        # Extend Bool to and array
        done = self.extend_done(action.shape[0], done)
        
        # Compute target value
        _, current_value = self.network(state)
        _, next_value = self.network(next_state)

        # Comput Temporal Diffrence Target
        target_value = reward + self.gamma * next_value.detach().numpy() * (1 - done)
        target_value = torch.tensor(target_value, dtype=torch.float).to(self.device)

        # Compute losses
        critic_loss = F.mse_loss(current_value, target_value.detach())
        
        # Compute advantage
        advantage = (target_value - current_value).detach()
        
        # Compute policy loss with entropy regularization
        action_probs, _ = self.network(state)

        # Prepare index to target the Probs                             
        num_actions = action.shape[0]
        action_list = np.squeeze(action)                                  # Size(Batch + #Agents)

        # Batch size = 1
        if num_actions == 1 :
            log_probs = torch.log(action_probs[0, action_list])
        else:
            idx_rows = np.arange(len(action_list))                             # [0, ... , Total_samples]
            log_probs = torch.log(action_probs[idx_rows, action_list])
            
        actor_loss = -log_probs * advantage
        actor_loss = torch.mean(actor_loss)

        # Entropy regularization to encourage exploration
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
        
        # Total loss
        loss = critic_loss + actor_loss + 0.01 * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_item = loss.item()
        pi_loss_mean = torch.mean(actor_loss).detach().numpy()
        val_loss_mean = torch.mean( torch.squeeze(critic_loss) ).detach().numpy()
        td_target_mean = torch.mean(target_value).item()
        advantage_mean = torch.mean(advantage).item()

        total_iter_rewars = np.sum(reward)
        self.reward_records.append(total_iter_rewars) 
        self.TD_target_record.append(td_target_mean)
        self.pi_loss_record.append( pi_loss_mean )
        self.val_loss_record.append( val_loss_mean )
        self.loss_record.append( loss_item )
        self.advantage_record.append( advantage_mean )

        # self.cumulative_reward_record.append(self.cumulative_reward_record[-1] + reward )
        if done_flag :
            self.cumulative_reward_record.append( total_iter_rewars )

        # Visualization
        print()
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        print("     Pi Loss = {}  Val. Loss = {} Loss = {} ".format(pi_loss_mean, val_loss_mean, loss_item))
        print("___________________________________")
        self.global_steps_T += 1

        if vis_flag :
            print("--------- Training Function ---------")
            print("Input States Shape  = ", state.shape)
            print("Input Actions Shape = ", action.shape)
            print("Input Rewards       = ", reward)
            print()
            print("TD Target Shape    = ", td_target_mean)
            print("Actor Output Shape = ", action_probs.shape)
            print("Log( Probs ) Shape = ", log_probs.shape)
            print("Pi loss shape (record) = ", len(self.pi_loss_record))

    def train_step(self, state, action, reward, next_state, done, vis_flag=False):
        """
        Perform a single training step using Actor-Critic algorithm
        
        Args:
            state   (numpy): Current state
            action    (int): Action taken
            reward  (float): Reward received
            next_state (numpy): Next state
            done (bool): Episode termination flag
        """
        self.network.train()

        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        
        # Compute target value
        _, current_value = self.network(state)
        _, next_value = self.network(next_state)

        target_value = reward + self.gamma * next_value.detach().numpy() * (1 - done)
        # target_value = reward + self.gamma * next_value.detach().numpy() 
        target_value = torch.tensor(target_value, dtype=torch.float).to(self.device)

        # Compute losses
        critic_loss = F.mse_loss(current_value, target_value.detach())
        
        # Compute advantage
        advantage = (target_value - current_value).detach()
        
        # Compute policy loss with entropy regularization
        action_probs, _ = self.network(state)
        log_probs = torch.log(action_probs[0, action])
        actor_loss = -log_probs * advantage
        
        # Entropy regularization to encourage exploration
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
        
        # Total loss
        loss = critic_loss + actor_loss + 0.01 * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_item = loss.item()
        pi_loss_mean = torch.mean(actor_loss).detach().numpy()
        val_loss_mean = torch.mean( torch.squeeze(critic_loss) ).detach().numpy()
        td_target_mean = torch.mean(target_value).item()
        advantage_mean = torch.mean(advantage).item()

        total_iter_rewars = np.sum(reward)
        self.reward_records.append(total_iter_rewars) 
        self.TD_target_record.append(td_target_mean)
        self.pi_loss_record.append( pi_loss_mean )
        self.val_loss_record.append( val_loss_mean )
        self.loss_record.append( loss_item )
        self.advantage_record.append( advantage_mean )

        # self.cumulative_reward_record.append(self.cumulative_reward_record[-1] + reward )
        if done :
            self.cumulative_reward_record.append( total_iter_rewars )

        # Visualization
        print()
        print("Run episode {} with rewards {}".format(self.global_steps_T, total_iter_rewars))
        print("     Pi Loss = {}  Val. Loss = {} Loss = {} ".format(pi_loss_mean, val_loss_mean, loss_item))
        print("___________________________________")
        self.global_steps_T += 1

        if vis_flag :
            print("--------- Training Function ---------")
            print("Input States Shape  = ", state.shape)
            print("Input Actions Shape = ", action.shape)
            print("Input Rewards       = ", reward)
            print()
            print("TD Target Shape    = ", td_target_mean)
            print("Actor Output Shape = ", action_probs.shape)
            print("Log( Probs ) Shape = ", log_probs.shape)
            print("Pi loss shape (record) = ", len(self.pi_loss_record))


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



    def plot_training(self, episodes):
        '''
            Plot in a row:
                (1) Reward
                (2) Pi. Loss
                (3) Val. Loss
        '''
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 3, figsize=(12, 4))

        # First plot
        axes[0, 0].plot(self.cumulative_reward_record, 'r')
        axes[0, 0].set_title("Rewards by Episode " + str(episodes))

        axes[1, 0].plot(self.reward_records, 'r')
        axes[1, 0].set_title("Rewards by Steps " + str(self.global_steps_T))

        # Second plot
        axes[0, 1].plot(self.pi_loss_record, 'g')
        axes[0, 1].set_title("Pi. loss")

        # Third plot
        axes[0, 2].plot(self.val_loss_record, 'b')
        axes[0, 2].set_title("Vsal. loss")

         
        axes[1, 1].plot(self.TD_target_record, 'b')
        axes[1, 1].set_title("TD Target")

        axes[1, 2].plot(self.advantage_record, 'g')
        axes[1, 2].set_title("Advange")

        # Adjust layout
        for axe in axes:
            for ax in axe :
                # ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_training_oneRow(self):
        '''
            Plot in a row:
                (1) Reward
                (2) Pi. Loss
                (3) Val. Loss
        '''
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))

        # First plot
        axes[0].plot(self.reward_records, 'r')
        axes[0].set_title("Rewards by step " + str(self.global_steps_T))

        # Second plot
        axes[1].plot(self.pi_loss_record, 'g')
        axes[1].set_title("Pi. loss")

        # Third plot
        axes[2].plot(self.val_loss_record, 'b')
        axes[2].set_title("Vsal. loss")

        # 
        axes[3].plot(self.TD_target_record, 'r')
        axes[3].set_title("TD Target")

        # Adjust layout
        for ax in axes:
            # ax.set_aspect('equal')
            ax.grid(True)

        plt.tight_layout()
        plt.show()