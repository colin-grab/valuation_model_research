# -*- coding: utf-8 -*-
"""
@author: Colin Grab

"""

import numpy as np
import pandas as pd
import math

class base_policy():
    
    def __init__(self,target_allocation, name, tc, deviation_treshold = 0.0, reward_type = 'log'):
        
        self.initial_value = 1.0
        
        #for now also don't allow short (can adjust for later)
        target_allocation = np.abs(target_allocation)  
    
        #if target_allocation does not sum up to 1 -> normalize such that sum=1
        if np.sum(target_allocation) != 1:
            target_allocation = target_allocation/(np.sum(target_allocation))
            print(f'Input target allocation for base_policy {name} is rescaled ')
            
            #could also just fill up one of them, or add 
            # e.g. target_allocation[-1] += (1-np.sum(target_allocation))

        self.target_allocation = target_allocation
        
        self.deviation_treshold = deviation_treshold
        self.key_name = name
        
        self.position = self.target_allocation
        self.adjusted_position = pd.Series(self.target_allocation)
        
        self.value = self.initial_value
        self.tc = tc
        self.step = 0
        self.reward_type = reward_type
        
    def update_state(self,asset_return_period):
        #for the moment -> rename to align them (inside env -> insure correct names)
        
        # if deviation bigger then treshold -> set position back to target allocation
        if  np.sum(abs(self.target_allocation - self.adjusted_position)) > self.deviation_treshold:
            #print('REBALANCE')
            self.position = self.target_allocation
        # if not -> position for new period remains as is (i.e. adjusted position)
        else:
            self.position = self.adjusted_position
            
        #compute the return which consists of cost of changing from (adjusted to position) + change over period        
        period_return = np.dot(self.position,asset_return_period) * ( 1 - self.tc *(self.position - self.adjusted_position).abs().sum())
        
        reward = np.log(period_return)
        
        #update total worth/value of position -> value * (1+return)
        self.old_value = self.value
        self.new_value = self.value * (period_return)
        
        self.worth_change = self.new_value-self.value

        #
        self.value = self.new_value
        
        #update adjusted position (how position set at t changed over period)
        self.adjusted_position = (self.position * asset_return_period) / np.dot(self.position, asset_return_period)
        
        self.step +=1
        
        
        if self.reward_type == 'change':
            if math.isnan(self.worth_change):
                self.worth_change = 0

            return self.worth_change, self.value

        #to insure there is never a nan as reward        
        if math.isnan(reward):
            reward = 0
        if np.isnan(reward):
            reward = 0
            
        return reward,self.value

    def length_(self):
        return len(self.target_allocation)
    
    def info(self):
        
        info_dict = {'Name':self.key_name,
                     'Target_allocation' : self.target_allocation,
                     'Current_allocation' : self.adjusted_position.values,
                     'Initial_value':self.initial_value,
                     'Current_value':self.value,
                     'Allowed_deviation':self.deviation_treshold,
                     'Current_deviation':np.sum(abs(self.target_allocation - self.adjusted_position))}
        
        return info_dict
    
 