# -*- coding: utf-8 -*-
"""

DEFINE VALUATION MODEL CLASS

@author: Colin Grab

"""

from imports import *
from utils.replay_buffers import general_ExperienceReplayBuffer


tf.config.run_functions_eagerly(True)

class valuation_model():

    def __init__(self, shape_input,
                 n_version,valuation_network,
                 n_steps = 1,gammas = [0.999],
                 aux_keys = {},
                 reward_type = 'log',
                 asset_type ='crypto',
                 distributional_aux = False,
                 n_atoms = 51,
                 v_min = -1, v_max = 0.5,
                 #learining parameters
                 learning_rate = 0.0002,
                 #sizes of replay buffer and batches
                 memory_size = int(1e6),tb_interval = 25,checkpoint_interval =100,
                 batch_size = 24,
                 #update speed of target network
                 tau=0.005,
                 #parameters for experience buffer, to update priorities
                 alpha_start = 0.75, alpha_end = 0.0, learning_steps_to_alpha_decay = 2e4,
                 beta_start = 0.25, beta_end = 1.0, learning_steps_to_beta_decay = 2e4,
                 #
                 env_config=None,output_path = None,support_file_path = None):
        
        self.env_config  = env_config        
        self.output_path = output_path
        self.support_file_path = support_file_path
        
        self.reward_type = reward_type
        self.asset_type = asset_type
        
        self.n_steps = n_steps 
        self.gammas = gammas
        self.n_gammas = len(self.gammas)
                
        self.aux_keys = aux_keys
        
        #distributional_aux = True or False  -> estimate a value function or estimate distributional value function 
        self.aux_dist = distributional_aux
        
        #Hyperparams for distributonal aux cases for single single support case
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = tf.linspace(start = tf.cast(self.v_min,tf.float32), stop = self.v_max, num = self.n_atoms)
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)        

        #to be able to use different supports for different gammas (as deppending on gamma different values more likely-> make support as dictionary                           
        self._initialize_support()
        
        #Hyperparams for learning &target network update speed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        
        #learning step counter
        self.learn_step_counter = 1
        self.tb_interval = tb_interval   #how often to write to tensorboard log
        self.checkpoint_interval = checkpoint_interval   #how often to save a checkpoint
        
        #Hyperparams of ReplayBuffer
        self.alpha_start =alpha_start
        self.alpha_end = alpha_end
        self.learning_steps_to_alpha_decay = learning_steps_to_alpha_decay
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_steps_to_beta_decay = learning_steps_to_beta_decay        
        self.memory_size = memory_size
        
        #Initialize the buffer
        self.buffer = general_ExperienceReplayBuffer(capacity = memory_size,
                                                     gammas = True,aux_task = True,
                                                     alpha =self.alpha_start,beta = self.beta_start)
        self.transitions = []

        ''' Inititalize 2 networks 1 for evaluation and one for estimation of next step '''   
        self.n_version = n_version
        self.shape_input = shape_input
        self.valuation_network = valuation_network
    
        self.clip_threshold =1.0
        
        # CREATE OUTPUT FILE NAMES&STRUCTURE (for consistency) and initialize other necessary paths etc.
        self.create_file_names_and_paths()

        #initialize neural net
        self.initialize_valuation_model()
        #after initialization set weights of target equal:
        self.soft_update_target_network(tau=1)        
        #initialize checkpoints 
        self._initialize_checkpoints()
        
        #at end of initialization write out the initialization values to a config (can be used later to initiate model but also to keep track of the runs)
        c = self.get_agent_config()
        

                
    def initialize_valuation_model(self):
        """
        FUNCTION TO INITIALIZE THE VALUATION NEURAL NETWORK AS WELL AS A TARGET NETWORK
        (NOTE: for more variation in network structure & additional input probably good idea to put
         input argument in dict to only need to change inputs and not code inside model class here)
        Returns
        -------
        None.

        """
        
        #valuation network
        val_model = self.valuation_network(shape_input = self.shape_input,
                                           aux_keys = self.aux_keys,
                                           aux_dist = self.aux_dist,
                                           n_gammas = self.n_gammas,
                                           n_atoms = self.n_atoms)
        self.val_model = val_model.model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.val_model.compile(optimizer=self.optimizer)

        #target network
        target_val_model =self.valuation_network(shape_input = self.shape_input,
                                                  aux_keys = self.aux_keys,
                                                  aux_dist = self.aux_dist,
                                                  n_gammas = self.n_gammas,
                                                  n_atoms = self.n_atoms)
        self.target_val_model = target_val_model.model()
        self.target_optimizer = Adam(learning_rate=self.learning_rate)
        self.target_val_model.compile(optimizer=self.target_optimizer)



    def _initialize_checkpoints(self,):   
        """
        HELPER FUNCTION TO INIT CHECKPOINT MANAGER FOR VALUE AND TARGET NETWORK

        Returns
        -------
        None.

        """             
        ## initialize checkpoint (to save and restore state of model and optimizer)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              optimizer=self.optimizer, 
                                              model=self.val_model)
        ## and a checkpoint manager 
        self.manager = tf.train.CheckpointManager(self.checkpoint, 
                                                  directory=os.path.join(self.checkpoint_directory,'chkpt_manager'), 
                                                  max_to_keep=3)
        
        ## same for target network
        self.target_checkpoint =  tf.train.Checkpoint(step=tf.Variable(1),
                                                      optimizer=self.target_optimizer, 
                                                      model=self.target_val_model)
        ## and a checkpoint manager 
        self.target_manager = tf.train.CheckpointManager(self.target_checkpoint, 
                                                         directory=os.path.join(self.checkpoint_directory,'chkpt_manager_target'), 
                                                         max_to_keep=3)  
        pass

    ### LEARNING FUNCTION
    def learn(self):
        """
        FUNCTION PERFORMING A LEARNING STEP
        
        SAMPLES BATCH OF EXPERIENCES FROM EXPERIENCE REPLAY BUFFER
        CALCULATES ESTIMATES AND TARGET ESTIMATES AND ESTIMATES LOSS OVER BATCH
        
        UPDATES SAMPLED SAMPLES IN BUFFER
        
        CALCULATES GRADIENTS OF LOSS AND UPDATES NETWORK WEIGHTS
        
        WRITES OUT LOGS FOR TENSORBOARD
        UPDATES TARGET NETWORK AND BUFFER PARAMETER
        
        Returns
        -------
        None.

        """        

        #if there are not enough samples in buffer -> no learning yet
        if len(self.buffer) < self.batch_size:
            return   
        
        ### sample a batch from the buffer        
        indices, weights, states, dones, next_states,aux_worth_values,aux_rewards = self.buffer.sample(self.batch_size)

        weights = tf.convert_to_tensor(weights, tf.float32)
        dones = tf.convert_to_tensor(dones)
        dones_ = tf.expand_dims(tf.where(dones, 1.0, 0.0),axis = 1)
        
        #reformat auxiliary rewards
        if self.n_steps == 1:
            aux_r = tf.expand_dims(tf.convert_to_tensor(aux_rewards,tf.float32),axis=-1) #Aux_task,Batch,1
            aux_r = tf.repeat(aux_r, self.n_gammas, axis=-1) #Aux_task,Batch,Gammas #probably not needed and tf checks automatically, but for clarity
        else:
            aux_r = tf.convert_to_tensor(aux_rewards,tf.float32)   # Aux_task, Batch,Gammas
        
        #format states
        states = tf.convert_to_tensor([state_x.values for state_x in states])
        next_states = tf.convert_to_tensor([state_x.values for state_x in next_states])
        
        #format worth 
        w_s = tf.expand_dims(tf.convert_to_tensor(aux_worth_values[0],tf.float32),axis=-1)
        w_ns = tf.expand_dims(tf.convert_to_tensor(aux_worth_values[1],tf.float32),axis=-1)
        
        #DIFFERENCE: FOR CHANGE CASE NEED TO INCORPORATE WORTH IN CALC..
        #-> update for V learning -> scale by worth
        #-> update for D learning -> need to adjust projection to include shift and scaling by worth.
        
        with tf.GradientTape() as tape:
            
            #create estimates using the model networks
            x_s,  aux_val_s= self.val_model(states)            
            x_target_ns,aux_target_ns= self.target_val_model(next_states)   
            
            #convert to tensor 
            v_s = tf.convert_to_tensor(aux_val_s)   # TENSOR SHAPE AUX_TASK, BATCH,GAMMA,ATOMS
            v_ns = tf.convert_to_tensor(aux_target_ns) # TENSOR SHAPE AUX_TASK, BATCH,GAMMA,ATOMS
            
            
            # IF DISTRIBUTIONAL VERSION
            if self.aux_dist:
                
                #permute to have order TASK,GAMMA,BATCH,ATOMS
                v_s = tf.transpose(v_s, perm=[0,2, 1, 3]) #TASK,GAMMA,BATCH,ATOMS
                v_ns = tf.transpose(v_ns, perm=[0,2, 1, 3]) #TASK,GAMMA,BATCH,ATOMS
                aux_r = tf.transpose(aux_r, perm=[0, 2, 1]) #TASK,GAMMA,BATCH

                #outloop over aux_task, inner loop over gamma
                #input to bin_migration_function: (Batch,Atoms) or (Batch,1)
                aux_entros = [[self.categorical_bin_migration(rewards = r,
                                                              est_probs_value_s = vs,
                                                              est_probs_target_ns = vns,
                                                              worth_s =w_si,
                                                              worth_ns = w_nsi,
                                                              dones = dones_,
                                                              gamma = g)
                               for vs,vns,r,g in zip(v_est_s,v_est_ns,rew,self.gammas)]
                              for v_est_s,v_est_ns,rew,w_si,w_nsi in zip(v_s,v_ns,aux_r,w_s,w_ns)]

                aux_entros_t = tf.convert_to_tensor(aux_entros)  # shape Task,Gammas,Batch

                # get td error per sample batch -> #use mean or sum (sum might be better to prioritize samples)
                td_errors = tf.reduce_sum(tf.abs(aux_entros_t),axis=[0,1])
                
                #apply priority weights on batch level (weighted average)
                gamma_mean_loss = tf.reduce_sum(aux_entros_t*weights,axis=-1)/tf.reduce_sum(weights)
                
                #average auxiliary loss => mean over gamma level
                summed_aux_losses = tf.reduce_mean(gamma_mean_loss,axis=-1)
                
                #total loss
                aux_loss = tf.reduce_mean(summed_aux_losses)
            
            # ELSE ESTIMATE EXPECTED RETURN CASE
            else:
                
                gamma_tensor = tf.expand_dims(tf.convert_to_tensor(self.gammas),axis=0)
                weights = tf.expand_dims(weights,axis=-1)
                summed_weights = tf.reduce_sum(weights)
                
                if self.reward_type != 'log':
                    #multiply estimates by worth:
                    v_s = v_s * w_s     #(AUX TASKS, BATCH , GAMMA)
                    v_ns = v_ns * w_ns  #(AUX TASKS, BATCH , GAMMA)
                
                target = aux_r + v_ns * gamma_tensor**self.n_steps * (1-dones_) #(AUX TASKS, BATCH , GAMMA)
                
                #get td_errors per sample in batch for priority   
                td_errors = tf.reduce_sum(tf.abs(target-v_s),axis=[0,2])  #(BATCH,)

                #calculate mse
                mse_ = 0.5 * tf.square(target-v_s) #(AUX TASKS, BATCH , GAMMA)
                #apply priority weights on batch dim/level (weighted average)
                gamma_mean_loss = tf.reduce_sum(mse_*weights,axis=1)/tf.reduce_sum(weights) #(AUX TASK,GAMMAS)
                
                #average auxiliary loss => mean over gamma level
                summed_aux_losses = tf.reduce_mean(gamma_mean_loss,axis=-1) #(AUX TASK,)
                
                #total loss
                aux_loss = tf.reduce_mean(summed_aux_losses)


            #update the priorities in buffer    
            for idx, error in zip(indices,td_errors):
                if isinstance(error, tf.Tensor):
                    error = error.numpy()
                self.buffer.update_priority(idx,error)
            
            #define the total loss 
            total_loss = aux_loss    

        #Calculate GRADIENTS AND APPLY THEM
        params = self.val_model.trainable_variables
        grads = tape.gradient(total_loss, params)
        
        clipped_gradients, _ = tf.clip_by_global_norm(grads, self.clip_threshold)

        #catch nan occurance 
        if any(tf.reduce_any(tf.math.is_nan(g)) for g in grads):
            print("Encountered NaN gradients. Skipping gradient update.")
            
        else:
            #apply the gradients
            self.val_model.optimizer.apply_gradients(zip(clipped_gradients, params))
            
            #for models with Batchnormalization -> clear batch stats
            for layer in self.val_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.batch_stats = None
                    
            """ wirte out the loss and performance to track (only write out every k-th step"""
            
            if self.learn_step_counter % self.tb_interval == 0:
                tb_write_step =int(self.learn_step_counter/self.tb_interval)
            
                #write loss and tot loss (here same, but might be different for other variants)
                with self.train_writer.as_default():
                    with tf.name_scope("a) Loss"):
                        tf.summary.scalar("Tot. Loss", total_loss, step=tb_write_step)
                        tf.summary.scalar("Mean Aux Loss", aux_loss, step=tb_write_step)
                            
                    with tf.name_scope("d) Network " + self.file_name):
                        for par, grad in zip(params,grads):
                            tf.summary.histogram(par.name,par,step=tb_write_step)
                            tf.summary.histogram(par.name + '/gradient',grad,step=tb_write_step)  
                            
                    with tf.name_scope("e) Hyper Parameters"):
                        tf.summary.scalar("alpha", self.buffer.alpha, step=tb_write_step)
                        tf.summary.scalar("beta", self.buffer.beta, step=tb_write_step)
        
                    with tf.name_scope("b) Aux Head Losses"):                        
                        for loss_,aux_name_ in zip(summed_aux_losses,self.aux_keys):
                            tf.summary.scalar(f"Aux_loss_{aux_name_}", loss_, step=tb_write_step) 
                      
                        
                      
            """ at end of learning step update hparameters + save"""
            self.soft_update_target_network()        
            
            #save checkpoints periodically
            self.checkpoint.step.assign_add(1)
            if int(self.checkpoint.step) % self.checkpoint_interval == 0:
                save_path = self.manager.save()
                #print(f'saved to {save_path}')
                save_target_path = self.target_manager.save()
                #save parameters:
                self.save_info_to_resume(checkpoint=True)
            
            self.learn_step_counter += 1       
            self.update_buffer_params() 

        
    ### helper functions    
    @tf.function
    def categorical_bin_migration(self, rewards,
                                  est_probs_value_s,
                                  est_probs_target_ns,
                                  worth_s,worth_ns,
                                  dones, gamma):
        """
        FUNCTION TO CALCULATE CROSS ENTROPY LOSS FOR CATEORICAL DISTRIBUTIONAL VARIANT
        (LOSS BETWEEN ESTIMATED DISTRIBUTION FOR CURRENT STATE AND NEXT_STATE)

        Parameters
        ----------
        rewards : TF TENSOR
            REWARDS (BATCH,) .
        est_probs_value_s : TF TENSOR
            ESTIMATED PROBABILITIES OVER SUPPORT (BATCH,N_ATOMS).
        est_probs_target_ns : TF TENSOR
            ESTIMATED PROBABILITIES OVER SUPPORT (BATCH,N_ATOMS).
        worth_s : TF TENSOR
            WORTH AT STATE S (BATCH,1).
        worth_ns : TF TENSOR
            WORTH AT NEXT STATE (BATCH,1).
        dones : TENSOR
            DONE (1 or 0) TO FLAG SHOW END OF EPISODE (BATCH,1).
        gamma : FLOAT, optional
            GAMMA VALUE USED IN TARGT CALC. The default is None WHICH WILL DEFAULT TO AGENT MAIN GAMMA.

        Returns
        -------
        cross_entropies : TF TENSOR
            Estimated  losses.

        """

        #use support dict to get correct support for given gamma value:
        support_ = self.dict_support[gamma]
        v_min = self.dict_min[gamma]
        v_max = self.dict_max[gamma]
        support_ = tf.expand_dims(support_,axis = 0)
        delta = (v_max - v_min) / (self.n_atoms - 1) 
        
        #format rewards to (BATCH,1)
        rewards_ = tf.expand_dims(tf.cast(rewards, tf.float32),axis = 1) 
        
        #take log of est prob values
        log_state_prob = tf.math.log(est_probs_value_s+1e-5)

        #depending on reward_type scale projection by worth or not
        if self.reward_type != 'log':
            #bellman equation step projection on support but with additional scaling for worth
            Tz = rewards_/worth_s + worth_ns/worth_s * gamma ** self.n_steps * support_ *(1 - dones)
        else:
            Tz = rewards_ + gamma**self.n_steps * support_ *(1 - dones)

        #clip by min and max
        Tz = tf.clip_by_value(Tz, clip_value_min = v_min, clip_value_max=v_max)
        #find 'location'
        b = (Tz - v_min)/delta
        
        #in very very rare cases for n_atoms 51 there was b with an entry 50.00000004 which will give out of bounce (as upper=51 + offset ->index from 0 to 51 -> will be outside range)
        #thus insure there are only valid indices 
        b = tf.clip_by_value(b, clip_value_min = 0, clip_value_max=self.n_atoms-1)
        
        #find lower and upper bin
        lower = tf.math.floor(b)
        upper = tf.math.ceil(b)

        offset = tf.reshape(tf.range(self.batch_size, dtype=tf.int32),[-1,1])* self.n_atoms
        offset = tf.cast(offset, tf.float32)     #batch x 1

        lower_index = tf.cast(tf.reshape(lower +offset,[-1,1]),tf.int32)     # B x Number atoms
        upper_index = tf.cast(tf.reshape(upper +offset,[-1,1]),tf.int32)     # B x Number atoms

        #spread the probs (NOTE b-lower -> distance from lower to b: if this distance is short -> want more to be spred to up than down -> thus "switch" b-l for up and u-b for down)        
        upper_prob = tf.reshape(est_probs_target_ns * (b - lower) ,-1)
        lower_prob =  tf.reshape(est_probs_target_ns * (upper - b) ,-1)
        
        spread_probs_1 = tf.scatter_nd(indices =lower_index,updates =lower_prob, shape = tf.shape(upper_prob))
        spread_probs_2 = tf.scatter_nd(indices =upper_index,updates =upper_prob, shape = tf.shape(lower_prob))
        spreaded_probs = spread_probs_1 +spread_probs_2                
        
        estimated_projected_probs = tf.reshape(spreaded_probs,[self.batch_size,self.n_atoms])
        
        cross_entropies = - tf.reduce_sum(estimated_projected_probs * log_state_prob, axis =-1)
        
        return cross_entropies
                                               

    def soft_update_target_network(self, tau=None):
        """
        FUNCTION THAT UPDATES ALL WEIGHTS OF THE TARGET NETWORK TO BE A MIX OF VALUE NETWORK
        AND TARGET NETWORK (TAU) * VALUE NETWORK + (1-TAU) * TARGET NETWORK
        
        THE HIGHER TAU THE CLOSER TARGET NETWORK IS TO VALUE NETWORK

        Parameters
        ----------
        tau : FLOAT, optional
            PARAMETER CONTROLLING HOW MUCH UPDATE IS USED. The default is None. 
            (defaults to input parameter tau, tau=1 equals identical copy)

        Returns
        -------
        None.

        """
        """ function to soft update the target network"""
        if tau is None:
            tau = self.tau
        
        #update the target weights according to tau * val_net + (1-tau) * target_net
        weights = []
        targets = self.target_val_model.weights
        for i, weight in enumerate(self.val_model.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_val_model.set_weights(weights)

    def update_buffer_params(self):
        """
        FUNCTION THAT UPDATES ALPHA AND BETA PARAMETERS OF THE EXPERIENCE REPLAY BUFFER 
        USES LINEAR DECAY DEPENDING ON HOW MANY LEARNING STEPS THE MODEL HAS PERFORMED

        Returns
        -------
        None.

        """
        # update alpha and beta parameters with simple linear decay:
        alpha = max(self.alpha_end , self.alpha_start - self.learn_step_counter / self.learning_steps_to_alpha_decay)
        beta = min(self.beta_end , self.beta_start + self.learn_step_counter / self.learning_steps_to_beta_decay)
        #update the params in the buffer
        self.buffer.alpha = alpha
        self.buffer.beta = beta        
    
        
    def process_experience_from_env(self,state,next_state, done, aux_rewards = None,value_dict = None):
        """
        FUNCTION THAT TAKES THE FEEBACK FROM ENVIRONMENT AS INPUT, FORMATS THEM AS NECESSARY 
            - for n-steps=1: simple one-step transition to add to buffer 
            - for n-steps >1 create transitions with n-step returns: sum_0^n gamma^i r_(t+i); NS euqal obs after n-steps
        AND HANDLES ADDITION TO BUFFER
        NOTE: uses global class variables n_steps, gammas, and transitions (which is just a helper list)
        Parameters
        ----------
        state : DF OR TUPLE
            REPRESENTATION OF STATE.
        next_state : DF OR TUPLE
            REPRESENTATION OF NEXT STATE.
        done : BOOL
            FLAG IF END OF EPISODE.
        aux_rewards : LIST OF FLOAT, optional
            AUXILIARY REWARDS OF PERIOD. The default is None.
        value_dict : TUPLE, optional
            TUPLE WITH 2 ELEMENTS WHERE FIRST WORTH AT S, SECOND WORTH AT NS 
        Returns
        -------
        None.
    
        """        
        #case n_steps = 1 -> directly append experience to buffer, multiple gamma cases can be handled in train fct
        if self.n_steps == 1:
            
            transition_dictionary = {'state':state,
                                     'done':done,
                                     'next state':next_state,
                                     'aux_reward':aux_rewards,
                                     'aux_worth':value_dict}
            self.buffer.append(transition_dictionary)
            return
        
        exp = (state, done, next_state,aux_rewards,value_dict)
        self.transitions.append(exp)
        
        if len(self.transitions) >= self.n_steps:
            self.transitions = self.transitions[-self.n_steps:]   #to avoid growing of helper list
            
            (s, d, ns,aux_r,value_dict) =self.transitions[0]            
            batch = self.transitions
            
            if len(batch) == self.n_steps:
    
                mg_r_sum = [
                    sum(np.array(list(transition[3].values()))*gamma_i ** j_step for j_step,transition in enumerate(batch) )
                    for gamma_i in self.gammas
                ]                    
                aux_r_sum = [[gamma_r[i] for gamma_r in mg_r_sum] for i,keys_ in enumerate(self.aux_keys)]
                aux_ret_ = dict(zip(self.aux_keys,aux_r_sum))                    
                
                #extract last experience in trajectory
                l_s,l_done, l_ns,l_aux,l_value_dict = batch[-1]
                

                transition_dictionary = {'state':s,
                                         'done':l_done,
                                         'next state':l_ns,
                                         'aux_reward':aux_ret_,
                                         'aux_worth':(value_dict[0],l_value_dict[1])}
                self.buffer.append(transition_dictionary)
                if l_done:
                    self.transitions=[]
        return    

    def _initialize_support(self):
        """
        HELPER FUNCTION TO INITIALIZE DICTIONARY HOLDING TF SUPPORT FOR EACH GAMMA
        (RANGE WAS DEDUCTED FROM OBSERVED RETURNS ON RANDOM RANGES OF TRAINING DATA)

        Returns
        -------
        None.

        """
        self.dict_support,self.dict_min,self.dict_max = {},{},{}

        if self.support_file_path:
            
            self.support_bounds = pd.read_csv(self.support_file_path,index_col=0)

            for gamma_input in self.gammas:
                try:
                    bound = self.support_bounds.loc[self.support_bounds.gamma ==float(gamma_input),'even_bound'].values[0]
                except:
                    print(f'FOR FOLLOWING GAMMA INPUT: {gamma_input} bounds are not pre-defined')
                    continue
                
                self.dict_support[gamma_input] =  tf.linspace(start = tf.cast(-bound,tf.float32), stop =bound, num = self.n_atoms)
                self.dict_min[gamma_input],self.dict_max[gamma_input] = -bound, bound
        
        #alternative could also just provide bounds:
        else:
            gamma_bounds = {0.999:0.05,
                            0.9975:0.03,
                            0.995:0.025,
                            0.99:0.02,
                            0.98:0.0175,
                            0.975:0.015,
                            0.97:0.0125,
                            0.96:0.01,
                            0.95:0.0075,
                            0.94:0.007,
                            0.93:0.0065,
                            0.925:0.006,
                            0.91:0.0055,
                            0.9:0.005,
                            0.85:0.0045,
                            0.8:0.004,
                            0.75:0.0035,
                            0.7:0.003,
                            0.65:0.003,
                            0.6:0.0025,
                            0.55:0.0025,
                            0.5:0.002}
            
            for gamma_input in self.gammas:
                bound = gamma_bounds.get(float(gamma_input))
                self.dict_support[gamma_input] =  tf.linspace(start = tf.cast(-bound,tf.float32), stop =bound, num = self.n_atoms)
                self.dict_min[gamma_input],self.dict_max[gamma_input] = -bound, bound
            

    """ OTHER HELPER FUNCTIONS TO SAVE INFOS,CONFIGS, FULL MODELS """
                        
    def save_info_to_resume(self,checkpoint=False):
        """
        SIMPLE HELPER FUNCTION TO SAVE RELEVANT CURRENT HYPERPARAMS TO ENABLE RESUMING AT LATER POINT


        Parameters
        ----------
        checkpoint : BOOL, optional
            SPECIFIES IF FUNCTION IS CALLED IN CONTEXT OF CHECKPOINT SAVING. The default is False.
            (WILL SAVE TO DIFFERENT SUBFOLDER IN OUTPUT DEPENDING IF PART OF REPEATED CHECKPOINTS)
        Returns
        -------
        None.

        """

        if checkpoint:
            self.parameters_info_ = os.path.join(self.checkpoint_directory,f'current_parameters_{self.file_name}')  
        else:
            #save the configuration to the output folder
            self.parameters_info_ = os.path.join(self.output_directory,f'model_weights/parameters_ls_{self.learn_step_counter}_{self.file_name}') 
        
        parameters_dict = {'alpha':float(self.buffer.alpha),
                           'beta':float(self.buffer.beta),
                           'max_priority':float(self.buffer.max_priority),
                           'learning_step':self.learn_step_counter}        
        
        try:
            json_data = json.dumps(parameters_dict)
            
        except Exception as e:
            print(f"An error occurred during JSON serialization: {str(e)}")        

        with open(self.parameters_info_, 'w',encoding='utf-8') as file:
            json.dump(parameters_dict, file)
        
    def save_model(self,both_models=True):
        """
        FUNCTION TO SAVE NEURAL NETWORKS OF MODEL

        Parameters
        ----------
        both_models : BOOL, optional
            Specify if both main model and target network should be saved. The default is True.
        Returns
        -------
        None.

        """
        
        model_info = f"_training_step{self.learn_step_counter}"        
        self.val_model.save(self.save_model_path  + '/model' +model_info,save_format='tf')
        
        if both_models:
            #save target model (only needed if want to resume training at certain step, to have same lag of target model weights)
            self.target_val_model.save(self.save_target_model_path  + '/model' + model_info,save_format='tf')
            

    def create_file_names_and_paths(self,):
        """
        SIMPLE HELPER FUNCTION TO INITIALIZE SOME FILE PATHS AND NAMES TO KEEP EVERYTHING CONSISTENT

        Returns
        -------
        None.

        """
        timestamp = time.strftime("%d%m_%Y_%H%M", time.localtime())
        self.timestamp = timestamp
        
        if self.output_path is None:
            #output directory simply relative to working directory     
            self.output_directory = f'output/{timestamp}/'        
        else:
            #output directory will be relative to specified path
            self.output_directory = f'{self.output_path}/output/{timestamp}/'
            
        if not os.path.exists(self.output_directory):
            # If the directory does not exist, create it
            os.makedirs(self.output_directory)
        else:
            print('DIRECTORY EXISTS')
        
        #directory for writer to write logs
        self.writer_dir = os.path.join(self.output_directory,'logs/')
        
        #set model name based on inputs  
        self.model = 'CGMV'
        if self.aux_dist:
            self.model = 'CDGMV'
        
        #create filename based on input caracters:
        self.file_name_pre = (f"{self.model.lower()}"
                          f"_ns{self.n_steps}_ng{len(self.gammas)}"
                          ) 
        self.file_name = self.file_name_pre + f'_{timestamp}'
        self.dir_name = os.path.join(self.writer_dir, self.file_name_pre)
        
        #initiate train_writer
        self.train_writer = tf.summary.create_file_writer(self.dir_name)
        
        #initialize paths to save tf models
        self.save_model_path = os.path.join(self.output_directory,'model_weights/main_model_' + self.file_name)
        #initialize paths to save tf models
        self.save_target_model_path = os.path.join(self.output_directory,'model_weights/target_model_' + self.file_name)
        
        #path for checkpoints
        self.checkpoint_directory = os.path.join(self.output_directory,'tf_checkpoints')
        self.chkpt_prefix = os.path.join(self.checkpoint_directory,"ckpt")  #then can use checkpoint.save(file_prefix=self.chkpt_prefix)

        
    def get_agent_config(self):
        """
        FUNCTION THAT GENERATES A DICTIONARY HOLDING ALL RELEVENT PARAMETERS AND HYPERPARAMETERS OF THE MODEL
        AND SAVES THE CONFIG TO OUTPUT FOLDER
        Returns
        -------
        agent_configuration : DICT
            DICTIONARY WITH ALL INFORMATION.

        """
        
        agent_configuration = {'model':self.model,
                               'n_version':self.n_version,
                               'n_steps':self.n_steps,
                               'gammas':self.gammas,
                               'distributional_aux':self.aux_dist,
                               'n_atoms': self.n_atoms,
                               'v_min': self.v_min, 
                               'v_max':self.v_max,
                               'learning_rate':self.learning_rate,
                               'memory_size':self.memory_size,
                               'batch_size':self.batch_size,
                               'tau':self.tau,
                               'alpha_start':self.alpha_start, 
                               'alpha_end':self.alpha_end, 
                               'learning_steps_to_alpha_decay':self.learning_steps_to_alpha_decay,
                               'beta_start':self.beta_start, 
                               'beta_end':self.beta_end, 
                               'learning_steps_to_beta_decay':self.learning_steps_to_beta_decay,
                               'file_name':self.file_name,
                               'timestamp':self.timestamp,
                               'nr_learn_steps':self.learn_step_counter,
                               'aux_keys':list(self.aux_keys),
                               'reward_type': self.reward_type,
                               'asset_type': self.asset_type
                               }
        
        #if there is an env config supplied -> add this
        if self.env_config:
            agent_configuration.update(self.env_config )
        self.agent_configuration = agent_configuration
        
        #save the configuration to the output folder
        config_filename = os.path.join(self.output_directory,f'config_{self.learn_step_counter}'+self.file_name) 
        self.config_filename = config_filename
        try:
            json_data = json.dumps(agent_configuration)
        except Exception as e:
            print(f"An error occurred during JSON serialization: {str(e)}")        

        with open(config_filename, 'w',encoding='utf-8') as file:
            json.dump(agent_configuration, file)
            print(f"Configuration at init saved to {config_filename}")

        return agent_configuration
    

    