# -*- coding: utf-8 -*-
"""

DEFINE EXPERIENCE REPLAY CLASS

@author: Colin Grab

"""


from imports import *
   
class general_ExperienceReplayBuffer():
    
    # initialize/construct the buffer
    def __init__(self, capacity,
                 uniform = False, alpha = 1.0, beta = 0.5, 
                 gammas = False, aux_task = False,
                 max_priority=1.001):
        """
        INITIALIZER OF CLASS OBJECT GENERAL EXPERIENCE REPLAY BUFFER
        NOTE:
            initialize a experience replay buffer consisting of two deque (list like objects), 
                - one holding dictionaries each representing an experience 
                  (switched to dictionaries to use get('key') simpler to read and less error prone)
                - the second deque holds the priorities given to the experiences.
            
        Parameters
        ----------
        capacity : INT
            MAX SAMPLES THE BUFFER CAN HOLD.
        uniform : BOOL, optional
            SHOULD THE BUFFER SAMPLE UNIFORMLY OR PRIORITIZED. The default is False.
        alpha : FLOAT, optional
            ALPHA PARAMETER OF BUFFER - AIDS IN CALC OF IS CORRECTION. The default is 1.0.
        beta : FlOAT, optional
            BETA PARAMETER OF BUFFER - AIDS IN CALC OF IS CORRECTION. The default is 0.5.
        gammas : FLOAT, optional
            SHOULD REWARDS FOR MULTIPLE GAMMAS BE CONSIDERED IN BUFFER. The default is False.
        aux_task : BOOL, optional
            SHOULD REWARDS FOR MULTIPLE AUXILIARY TASKS BE CONSIDERED IN BUFFER.. The default is False.
            
        Returns
        -------
        None.

        """
            
        self.capacity = capacity
        self.uniform = uniform
        #initialize the two deque
        self.experience_buffer = deque(maxlen = capacity)
        self.prios = deque(maxlen = capacity)
        
        # initialize alpha and beta
        self.alpha = alpha
        self.beta = beta 
        
        self.gamma = gammas
        self.aux_task = aux_task
        
        # initialize a max prio value to assign some depended prio to new samples
        self.max_priority = max_priority
        
    # get number of samples in buffer
    def __len__(self):
        return len(self.experience_buffer)
    
    # add/append new experiences to buffer
    def append(self, new_experience:dict):
        """
        Function to add new experience to buffer

        Parameters
        ----------
        new_experience : dict
            Dictionary holding.

        Returns
        -------
        None.

        """
        """ function to add a new experience to the buffer. It appends the experience to one deque and 
        adds at the same index a priority to the priority deque 
        Note: an experience shout be a dictionary."""
        self.experience_buffer.append(new_experience)
        self.prios.append(self.max_priority)
        
    # update the priorities/probabilities of specific sample in the buffer
    def update_priority(self, index,priority):
        """
        HELPER FUNCTION TO UPDATE PRIORITIES OF SAMPLES AT SPECIFIED INDEX
        -> if uniform buffer no updates done
        -> function will be called inside valuation model during learning step
        
        Parameters
        ----------
        index : INT
            index of sample in list for which prioritiy is updated.
        priority : FLOAT
            New priority to assign to sample with index .

        Returns
        -------
        None.

        """
        if self.uniform:
            return
            
        #update max prio if new prio is higher
        if priority > self.max_priority:
            self.max_priority = priority
        
        #update the priority of the sample 
        self.prios[index] = priority

    # sample multiple (number of batchsize) experiences from the buffer
    def sample(self, batch_size):
        """
        FUNCTION TO SAMPLE A BATCH OF SIZE batch_size from the BUFFER
        - DEPENDING IF UNIFORM OR PRIORITIZED BUFFER USES ACCORDING SAMPLING SCHEME

        Parameters
        ----------
        batch_size : INT
            Number of samples to sample.

        Returns
        -------
        indices : LIST
            LIST OF INDICES (INDICES IN BUFFER) SPECIFYING WHICH SAMPLE USED.
        weights : LIST
            LIST OF IMPORTANCE WEIGHTS TO CORRECT FOR SAMPLING.
        states : LIST
            LIST OF STATES.
        dones : LIST
            LIST OF BOOLEAN FLAGS IF AT END OF EPISODE.
        next_states : LIST
            LIST OF NEXT STATES.
        aux_worth_values : TUPLE
            FIRST ELEMENT LIST ALL AUX WORTH AT S, SECOND ELEMENT LIST ALL AUX WORTH AT NS.
        aux_rewards : LIST
            LIST WITH AUX REWARDS FOR ALL TASKS .

        """
                
        if self.uniform:
            #use same probabilities for all, and weights = 1
            probs = np.ones(self.__len__())
            weights = np.ones(self.__len__())            
        else:
            #turn priorities into a array and add some small constant to avoid division by zero
            pri = np.array(self.prios, dtype=np.float64) + 0.0001
            #use formula from paper to turn priorities into probabilities
            pri = pri ** self.alpha        
            probs = pri / pri.sum() #maybe better to use a softmax here, smoother?
            
            #compute importance sampling weights to adjust "importance of samples"     
            weights = (self.__len__() * probs) ** -self.beta
    
            # to insure weights are between 0,1, scale by max weight
            weights = weights / weights.max()
        
        #sample index using the calculated probabilities
        sample_index = random.choices(range(self.__len__()), weights = probs, k = batch_size)
        

        #use the sampled index to to get the actual samples from the deque
        indices = [idx for idx in sample_index]
        weights = [weights[idx] for idx in sample_index]
        
        states  = [self.experience_buffer[idx].get('state') for idx in sample_index]
        dones   =[self.experience_buffer[idx].get('done') for idx in sample_index]
        next_states = [self.experience_buffer[idx].get('next state') for idx in sample_index]
        
        aux_rewards = []
        if self.aux_task:
            aux_rewards = [[self.experience_buffer[idx].get('aux_reward').get(key) for idx in sample_index]
                           for key in self.experience_buffer[1].get('aux_reward').keys()]
        
        
        aux_worth_values = []
        if self.gamma:
            aux_worth_values_s = [[self.experience_buffer[idx].get('aux_worth')[0].get(key) for idx in sample_index]
                                for key in self.experience_buffer[1].get('aux_worth')[0].keys()]
            aux_worth_values_ns = [[self.experience_buffer[idx].get('aux_worth')[1].get(key) for idx in sample_index]
                                for key in self.experience_buffer[1].get('aux_worth')[0].keys()]
            aux_worth_values = (aux_worth_values_s,aux_worth_values_ns)
            
        return indices, weights, states,dones, next_states,aux_worth_values, aux_rewards 

    def switch_buffer_to_uniform(self):
        """
        Helper function: that just switches the uniform flag to true (such that one could swith the buffer to uniform at later stages in training)

        Returns
        -------
        None.

        """
        self.uniform = True
            