import torch
import numpy as np
import torch.nn as nn


'''
    Intend to be the file for Different networks to be tested with:
                1) a2c_test_v1_2.py training algorithm
                2) Actor and Critic are intedepdent networks

'''


class ActorNetwork_1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
            Actor Network # 1 (Model 3)
                
                Note: Should be use with Batch Data for 
                      Batchnoem layer

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorNetwork_1, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()                         
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
    

class ActorNetwork_2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
            Actor Network # 2 (Model 4)
                
                Note: Should be use with Batch Data for 
                      Batchnoem layer

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorNetwork_2, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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



class CriticNetwork_2(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        """
            Critic Network # 2 (Model 4)
                
                Note: Should be use with Batch Data for 
                       * Batchnorm layers

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(CriticNetwork_2, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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
    


class ActorNetwork_3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
            Actor Network # 2 (Model 5)
                
                Note: Should be use with Batch Data for 
                      Batchnoem layer

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(ActorNetwork_3, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 6*hidden_dim),
            nn.BatchNorm1d(6*hidden_dim),
            nn.ReLU(),
            nn.Linear(6*hidden_dim, 6*hidden_dim),
            nn.BatchNorm1d(6*hidden_dim),
            nn.ReLU(),
            nn.Linear(6*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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



class CriticNetwork_3(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        """
            Critic Network # 3 (Model 5)
                
                Note: Should be use with Batch Data for 
                       * Batchnorm layers

        Args:
            state_dim  (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Number of neurons in hidden layers
        """
        super(CriticNetwork_3, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 6*hidden_dim),
            nn.BatchNorm1d(6*hidden_dim),
            nn.ReLU(),
            nn.Linear(6*hidden_dim, 6*hidden_dim),
            nn.BatchNorm1d(6*hidden_dim),
            nn.ReLU(),
            nn.Linear(6*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.BatchNorm1d(4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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