# -*- coding: utf-8 -*-
"""

SIMPLE HELPER FUNCTION TO CREATE 'LIST OF BASE TASKS'
IN ITS OWN SCRIPT/FUNCTION FOR BETTER READABILITY

@author: Colin Grab
"""

  
import pandas as pd
import numpy as np
    
def create_format_base_policies(base_policies_input_list,included_alone = []):
    
    #can use something like: df = pd.DataFrame.from_records([ret_dict,target_allocation]) when pf are given as dict.
    
    #reformat the allocations to da dataframe an redistirbute
    data =  [(i.get('allocation'),i.get('name')) for i in base_policies_input_list]
    df = pd.DataFrame.from_records([dict_ for dict_, _ in data], index=[name for _, name in data])
    
    #standardize such that weights sum to 1
    df = df.abs()
    df = df.div(df.sum(axis=1), axis=0)
    df.fillna(0, inplace = True)  #fill nan by 0
    
    #sort (not necessary but clean
    df =df.sort_index(axis=1, level =[1])

    #add the ones that are not in a pf (just for completeness)
    if len(included_alone) != 0:
        for i in included_alone:
            df[i] = 0
            
    #df "formatted base_policies" -> reassign in policy list
    for alloc in base_policies_input_list:
        name_ = alloc.get('name')
        allocation_series = df.loc[name_]
        alloc['allocation'] = allocation_series

    return base_policies_input_list

def create_example_base_tasks():
    all_ = ['BTCUSDT','ETHUSDT','BNBUSDT','NEOUSDT','LTCUSDT']

    big_three = ['BTCUSDT','ETHUSDT','BNBUSDT']
    big_two = ['BTCUSDT','ETHUSDT']
    
    rando_1 = [0.3,0.3,0.15,0.15,0.1]
    rando_2 = [0.1,0.1,0.2,0.3,0.3]
    rando_3 = [0.5,0.25,0.15,0.05,0.05]
    rando_4 = [0.2,0.2,0.4,0.1,0.1]
    rando_5 = [0.1,0.3,0.3,0.15,0.15]
    rando_6 = [0.7,0.15,0.15,0,0]
    rando_7 = [0,0,0.4,0.3,0.3]
    included_alone =[]
    
    base_policies_input_list = [
        {'allocation':{key:1 for key in all_}, 'name':'eqw_crypto', 'deviation_treshold' : 0.05},
        {'allocation':{key:1 for key in big_three}, 'name':'eqw_big_three',  'deviation_treshold' : 0.04},
        {'allocation':{key:1 for key in big_two}, 'name':'eqw_big_two',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_1)}, 'name':'rando_1',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_2)}, 'name':'rando_2',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_3)}, 'name':'rando_3',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_4)}, 'name':'rando_4',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_5)}, 'name':'rando_5',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_6)}, 'name':'rando_6',  'deviation_treshold' : 0.04},
        {'allocation':{key:ratio for key,ratio in zip(all_,rando_7)}, 'name':'rando_7',  'deviation_treshold' : 0.04},        
        ]
    base_policies_input_list = create_format_base_policies(base_policies_input_list = base_policies_input_list,
                                                           included_alone =[])
    
    return base_policies_input_list



def create_example_base_tasks_etf():

    all_ = ['SPY', 'XLF','XLE','GLD','USO']   #SP500,Financials,Energy, Gold,Oil, #add Treasury? or Dow (DIA)

    sector_divers = ['XLF','XLE','GLD'] #Financial,Energy,Gold

    fin_energy = ['XLF','XLE']
    fin_emph = [65,35]

    commodities =['GLD','USO']  #Gold, US Oil,
    
    sp_gold = ['SPY','GLD']
    sp_gold_w = [75,25]
    
    all_indices = ['SPY','XLF','XLE','GLD','USO']
    all_indices_w = [40,15,15,15,15]
    all_indices_w2 = [45,15,5,20,15]
    rando = [0,25,25,25,25]
    
    
    base_policies_input_list = [
        {'allocation':{key:1 for key in all_}, 'name':'eqw', 'deviation_treshold' : 0.05},
        {'allocation':{key:1 for key in sector_divers}, 'name':'sector_divers', 'deviation_treshold' : 0.05},
        {'allocation':{key:1 for key in fin_energy}, 'name':'fin_energy', 'deviation_treshold' : 0.05},
        {'allocation':{key:1 for key in commodities}, 'name':'commodities', 'deviation_treshold' : 0.05},
        {'allocation':{key:ratio for key,ratio in zip(fin_energy,fin_emph)}, 'name':'fin_energy_w1',  'deviation_treshold' : 0.05},
        {'allocation':{key:ratio for key,ratio in zip(sp_gold,sp_gold_w)}, 'name':'sp_gold',  'deviation_treshold' : 0.05},
        {'allocation':{key:ratio for key,ratio in zip(all_indices,all_indices_w)}, 'name':'all_indices_w',  'deviation_treshold' : 0.05},
        {'allocation':{key:ratio for key,ratio in zip(all_indices,all_indices_w2)}, 'name':'all_indices_w2',  'deviation_treshold' : 0.05},
        {'allocation':{key:ratio for key,ratio in zip(all_indices,rando)}, 'name':'without_sp1',  'deviation_treshold' : 0.05},
        ]
    base_policies_input_list = create_format_base_policies(base_policies_input_list = base_policies_input_list,
                                                           included_alone =[])
    
    return base_policies_input_list
