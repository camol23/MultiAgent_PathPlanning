import torch
import numpy as np
import torch.nn as nn


'''
    Intend to be the file for Different networks to be tested with:
                1) PPO_v1.py training algorithm
                2) Actor and Critic are intedepdent networks

    Notes:
        **) Include new values in PPO_LIST and NETWORK_CLASS_LIST
'''

# 1: Split Nets
# 0: One Net
MODELS_PPO_LIST = [
    0,                          # Model 0 : 6 layers + 3 in actor and critic
    1,                          # Model 1 : 5 layers 
]


# ====================================== Model # 0 =================================================

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


# ================================================================================================

# ====================================== Model # 1 =================================================

class ActorNetwork_1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
            Actor Network # 1 
        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorNetwork_1, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),

            # Probability distribution over actions
            nn.Softmax(dim=-1)                              
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): Input state
        
        Returns:
            action_probs (torch.Tensor): Probability distribution over actions
        """
        
        # Actor
        relu = self.actor(state)        
        action_probs = self.output(relu)

        return action_probs



class CriticNetwork_1(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        """
            Critic Network # 1 (Model 3)
                
                Note: Should be use with Batch Data for 
                       * Batchnorm layers

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(CriticNetwork_1, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),                       
        )

        self.output = nn.Linear(hidden_dim, 1) 

    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): Input state
        
        Returns:
            state_value (torch.Tensor): Estimated state value
        """
        
        # Critic
        relu = self.critic(state)
        values = self.output(relu)

        return values
    

# ================================================================================================








#### Networks

NETWORK_CLASS_LIST = [ActorCritic, (ActorNetwork_1, CriticNetwork_1)]