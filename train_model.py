# -*- coding: utf-8 -*-
"""

Example training script

@author: Colin Grab
"""


# import os
# dir_path = SET_TO_LOCAL_DIRECTORY
# os.chdir(dir_path)


#%% Import Modules,Classes,Util Functions
from imports import *

#import buffer & environment, base task
from utils.replay_buffers import general_ExperienceReplayBuffer
from utils.environments import aux_env
from utils.base_policy_class import base_policy
from utils.create_base_policies_input_list import create_example_base_tasks

#import neural network
from networks.val_model_example_nn import cdg_example_nn
aux_valuation_network = cdg_example_nn

#import valuation model
from model.valuation_model import valuation_model



#%% set some input paths
input_config_file_path ='./input_config_file.txt'
support_file_path = './support_bounds.txt'
output_path =  None 



#%% ASSET SELECTION & CREATE BASE POLICIES INPUT LIST
assets = ['BTCUSDT','ETHUSDT','BNBUSDT','NEOUSDT','LTCUSDT']
base_policies_input_list = create_example_base_tasks() #initiate base policies


#%% CONFIG FOR INPUT VARIABLES TO ENV AND MODEL

# load input variables:
with open(input_config_file_path) as file:
    input_config = json.load(file)

n_lags = input_config['n_lags']
decision_interval = input_config['decision_interval']
max_episode_length= input_config['max_episode_length']
min_episode_length= input_config['min_episode_length'] 
transaction_costs= input_config['transaction_costs']
n_version= input_config['n_version']
n_steps= input_config['n_steps']
gammas=  input_config['gammas']
distributional_aux=  input_config['distributional_aux']   
n_atoms= input_config['n_atoms']
v_min= input_config['v_min']
v_max= input_config['v_max']
memory_size= input_config['memory_size']
learning_rate= input_config['learning_rate']
batch_size =input_config['batch_size']
tau =input_config['tau']
reward_type = input_config['reward_type'] #'change' or 'log'


#%% for demonstration purpose load some sample data

input_data = pd.read_csv('./sample_toy_data.csv',index_col=0)



#%%initialize environment

aux_fin_env = aux_env(input_data =input_data,
                      assets = assets,
                      base_policies_input_list = base_policies_input_list,
                      n_lags = n_lags,
                      decision_interval =decision_interval, 
                      max_episode_length = max_episode_length, 
                      min_episode_length =min_episode_length,
                      transaction_costs = transaction_costs,
                      #single_case = 'BTCUSDT',
                      reward_type = reward_type
                      )
env_config = aux_fin_env._get_env_config()

#get info regarding shape of observation
obs,value_dict_s = aux_fin_env.reset()




#%% Define/Initialize model

model = valuation_model(shape_input = obs.shape,
                        n_version = n_version,
                        valuation_network = aux_valuation_network,
                        n_steps = n_steps, 
                        gammas =gammas,
                        aux_keys = aux_fin_env.aux_keys,
                        reward_type = reward_type,
                        distributional_aux = distributional_aux,
                        n_atoms = n_atoms, v_min = v_min, v_max = v_max,
                        memory_size = int(memory_size),
                        learning_rate = learning_rate,
                        batch_size = batch_size,
                        tau = tau,
                        env_config = env_config,
                        output_path=output_path,
                        support_file_path = support_file_path)

#save the model :  model.save_model()



#%% Example training loop // 

#can alternate between step and learn; or perform multiple learning steps and multiple step/epochs etc
#e.g. to avoid oversampling first epoch run some epochs first to fill buffer and start learning then.


nr_epochs_to_train = 1

for epoch_i in range(nr_epochs_to_train):
    
    #reset environment to start
    state, value_dict_s = aux_fin_env.reset()
    done = False
    
    #run over whole epoch
    while not done:
        #take step in environment
        next_state, done, info, aux_rewards, value_dict = aux_fin_env.aux_step()

        #process,format and add the experience to buffer
        model.process_experience_from_env(state =state,
                                          next_state=next_state,
                                          done = done,
                                          aux_rewards = aux_rewards,
                                          value_dict =(value_dict_s,value_dict))
        
        #update state 
        state = next_state
        value_dict_s = value_dict
        
        #perfom learning step 
        model.learn()
    print(f'end epoch {epoch_i}')

#to make one step pred: embedding_s, value_estimates_s = model.val_model(tf.convert_to_tensor([state.values]))
