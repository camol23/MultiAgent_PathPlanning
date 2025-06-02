import torch
import torch.nn as nn
import numpy as np
import os
import copy

'''
    Ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''


def save_model(model, optimizer, episode, reward_history, path="checkpoints", file_name="checkpoint_episode_", states=None, actions=None, rewards=None):
    """
        Save the DRL model, optimizer state, and training history
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            episode: Current episode number
            reward_history: List of rewards
            path: Directory to save the checkpoint
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    checkpoint = {
        'model_state_dict': copy.deepcopy( model.state_dict() ),
        'optimizer_state_dict': copy.deepcopy( optimizer.state_dict() ),
        'episode': episode,
        'reward_history': reward_history,
        'states_history' : states,                                      # Stored for buffer replay
        'actions_history' : actions,                                    # Stored for buffer replay
        'last_rewards' : rewards                                        # Stored for buffer replay
    }
    
    # checkpoint_path = os.path.join(path, f'checkpoint_episode_{episode}.pt')
    checkpoint_path = os.path.join(path, file_name + str(episode) + ".pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")



def load_model(model, optimizer, checkpoint_path):
    """
        Load a saved DRL model and its training state

        Args:
            model: PyTorch model instance to load weights into
            optimizer: Optimizer instance to load state into
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            model: Loaded model
            optimizer: Loaded optimizer
            episode: Episode number when checkpoint was saved
            reward_history: History of rewards
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    return (
        model,
        optimizer,
        checkpoint
    )

    # First version 
    # return (
    #     model,
    #     optimizer,
    #     checkpoint['episode'],
    #     checkpoint['reward_history']
    # )


# load model
def load_model_forTest(model, splitNets_flag, folder_number=None, checkpoint_name=''):
    '''
        Load pretrained models

            Args:
                1) folder_name (int) :  if None is loaded the last trained Model
                                        Otherwise select the that model

                2) checkpoint_name (str) : If a checkpoint is require is necessary specify its name
    
    '''

    # Count all files in a directory (Saved it when the training ends)
    if checkpoint_name == '' :

        folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_main'    
        if splitNets_flag :
            folder_path = folder_path + '/split'
        else:
            folder_path = folder_path + '/oneNet'

        if folder_number == None :
            checkpoint_files = [name for name in os.listdir(folder_path) ]
            num_files = len(checkpoint_files)
        else:
            num_files = folder_number

        if splitNets_flag :
            folder_path = folder_path + '/model_' + str(num_files)
            actor_path = folder_path + '/actor_v1/checkpoint_episode_' + str(num_files) + '.pt'
            critic_path = folder_path +  '/critic_v1/checkpoint_episode_' + str(num_files) + '.pt'

            model_path_str = actor_path
        else:
            actorCritic_path = folder_path + '/actorCritic_v1/checkpoint_episode_' + str(num_files) + '.pt'

            model_path_str = actorCritic_path

    else:
        # folder_number = 72
        # checkpoint_name = "/checkpoint_episode_174_reward_[2155.62004259].pt"

        if splitNets_flag :            
            checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1_splitNets'    
            folder_name = checkpoint_path + '/model_ppo_v1_test_' + str(folder_number)
            actor_path = folder_name + '/actor_ppo_v1' + checkpoint_name
            critic_path = folder_name + '/critic_ppo_v1' + checkpoint_name

            model_path_str = actor_path
        else:
            checkpoint_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_v1'
            folder_name = checkpoint_path + '/model_ppo_v1_test_' + str(folder_number)
            actorCritic_path = folder_name + '/actorCritic_v1' + checkpoint_name        

            model_path_str = actorCritic_path

    if splitNets_flag :
        model.actor_model, model.opt_actor, _ = load_model(model.actor_model, model.opt_actor, actor_path)
        model.critic_model, model.opt_critic, checkpoint_dict = load_model(model.critic_model, model.opt_critic, critic_path)
    else:
        model.actor_critic, model.optimizer, checkpoint_dict = load_model(model.actor_critic, model.optimizer, actorCritic_path)

    print()
    print("Model Loaded = ", model_path_str)
    print()

    return checkpoint_dict


def save_modelFromTraining(model, splitNets_flag, rewards_steps=None, data_list=None):
    '''
    
    '''
    # Count all files in a directory
    folder_path = './Guidance_controller/0_single_agent_v1/DRL/storage/checkpoints/model_ppo_main'   

    if splitNets_flag :
        folder_path = folder_path + '/split'
    else:
        folder_path = folder_path + '/oneNet'

    checkpoint_files = [name for name in os.listdir(folder_path) ]
    num_files = len(checkpoint_files)
    num_name = num_files + 1 

    if splitNets_flag :
        file_path = folder_path + '/model_' + str(num_name)

        actor_path = file_path + '/actor_v1'
        critic_path = file_path +  '/critic_v1'

        print("Saved as = ", actor_path + "_" + str(num_name) )
    else:
        file_path = folder_path + "/model_" + str(num_name)

        actorCritic_path = file_path + '/actorCritic_v1'

        print("Saved as = ", actorCritic_path + "_" + str(num_name) )
    

    # Store the model
    if splitNets_flag :
        save_model(model.actor_model, model.opt_actor, num_name, rewards_steps, actor_path)
        save_model(model.critic_model, model.opt_critic, num_name, rewards_steps, critic_path)    

    else:
        save_model(model.actor_critic, model.optimizer, num_name, rewards_steps, actorCritic_path)

    # Save the Log file
    if data_list != None:
        len_reward = rewards_steps.shape[0]
        if len_reward <= 10 :
            size = len_reward
        else:
            size = 10

        ave10_reward = np.mean( rewards_steps.squeeze()[-size:] ).item()
        max_reward = np.max( rewards_steps ).item()
        last_reward = rewards_steps[-1, -1]
        add_data = {
            'Max_reward' : max_reward,
            'Ave10_reward' : ave10_reward,
            'last_reward' : last_reward
        }
        
        log_name = "/log_" + str(num_name) + "_" + str(ave10_reward) + ".txt"
        training_results_txt(log_name, file_path, data_list, add_data)




def training_results_txt(model_name='/log.txt', folder_path=None, data_list=None, add_data=None):
    '''
        Create a .txt file to save Traning Config. from :
            log_data_list[ dict_1, ... ]  

        Args:
            add_data : Should be a Dictionary to add data to the Log file
    '''

    if folder_path != None :
        file_path = folder_path + model_name
        f = open(file_path, "w")

        if add_data != None :
            for key, value in add_data.items() :
                f.write('\n')
                f.write(key + ",    ," + str(value) + '\n')
                
        f.write('\n')
        f.write('\n')
        for data in data_list :
            for key, value in data.items() :                
                f.write(key + ",    ," + str(value) + '\n')
            
            f.write('\n')
        
        f.close()
        print("File '{}' created ".format(file_path))