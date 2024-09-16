# -*- coding: utf-8 -*-
"""

DEFINE ENVIRONMENT CLASS

@author: Colin Grab

"""

from imports import *
from utils.base_policy_class import base_policy

class aux_env():
    
    def __init__(self,input_data,assets,
                 base_policies_input_list=[],
                 n_lags = 1,decision_interval = 1,
                 max_episode_length = 10000, 
                 min_episode_length =500,
                 end_train_date=None,
                 transaction_costs = 0.002,
                 single_case = None,reward_type = 'log',
                 ): 
        """
        INITIALIZED AN ENVIRONMENT CLASS OBJECT 

        Parameters
        ----------
        input_data : DATAFRAME
            INPUT FEATURES FOR RELEVANT ASSETS SAVED AS PARTITIONED DATAFRAME, FILTERED TO ASSETS.
        assets : LIST
            LIST OF THE ASSETS TO USE.
        base_policies_input_list : LIST, optional
            LIST THAT HOLDS DICTIONARIES SPECIFYING BASE TASKS (assets,allocations,treshold). The default is [].
        n_lags : INT, optional
            NUMBER OF LAGS TO USE (ignored for current). The default is 1.
        decision_interval : INT, optional
            NUMBERS OF 1min interval per timestep. The default is 1.
        max_episode_length : INT, optional
            MAXIMUM LENGTH OF AN EPOCH. The default is 10000.
        min_episode_length : INT, optional
            MINIMUM LENGTH OF AN EPOCH. The default is 500.
        transaction_costs : FLOAT, optional
            VALUE OF TRANSACTION COSTS AS PROPPORTION. The default is 0.002.
        end_train_date : STRING, optional
            DATE TO SPECIFY END OF TRAINING PERIOD. The default is None.
        single_case : STRING, optional
            IF SYMBOL OF A ASSET IS GIVEN ENV WILL ONLY RETURN REWARD FOR THIS ASSET. The default is None.
        reward_type : STRING: 'log' or 'change', optional
            STRING SPECIFYING HOW REWARD IS CALCULATED. The default is 'log'.


        Returns
        -------
        None.

        """
        
        self.input_data = input_data
        
        self.assets = assets
        self.assets_ = [i +'USDT' if 'USDT' not in i else i for i in self.assets]
        
        self.single_case = single_case
            
        self.reward_type = reward_type
        self.base_policies_input_list = base_policies_input_list
        
        self.lags = n_lags
        self.decision_interval = decision_interval
        
        #set maximum length and minimum length of an episode
        self.max_episode_length = max_episode_length
        self.min_episode_length = min_episode_length
        self.end_train_date = end_train_date
        # assign transaction costs and a tc return factor
        self.tc = transaction_costs
        self.tc_factor = 1 - transaction_costs

        #initialize data
        self._initialize_data() 
        
        self.episode_count = -1
        
        # reset env
        self.reset()
        
        
    

    def _get_state(self):
        """
        HELPER FUNCTION THAT BASED ON CURRENT STEP IN EPPOCH (CURRENT BAR) SELECTS RELEVANT DATA/FEATURES FROM DATASET

        Returns
        -------
        DATAFRAME
            A DATAFRAME REPRESENTING CURRENT STATE FEATURES.

        """        
        
        data_for_t = self.data_.iloc[self.eppoch_bar -self.lags:self.eppoch_bar].copy()
        df_state = self._reformat_to_state(df = data_for_t) 
        
        df_state.index.name = data_for_t.index.values[-1]
        
        return df_state         

    def reset(self):
        """
        FUNCTION THAT RESETS THE ENVIRONMENT TO THE START OF AN EPOCH (ALSO RESETS EPOCH SPECIFIC VARIABLES TO START VALUES)


        Returns
        -------
        state : DF
            Data representing the current state (can be single df, but can also be formatted to for other data type, just need corresponding agent to accept it).
        value_dict_s : DICT
            Dictionary holding starting 'worth'.

        """

        #pick random start of a episode (max_index must be defined in prepare data)
        self.start_bar = random.randint(self.lags, self.max_index -self.min_episode_length)          

        #set current bar equal start bar
        self.bar = self.start_bar
        self.eppoch_bar = self.lags +1
        
        #given the sampled bar -> create the data df for the current eppoch
        self.end_bar = self.start_bar + self.max_episode_length + 2
    
        #get data
        self.data_ = self._get_data_for_eppoch()

        #set helper list transition (to collect experience for n_step cases)
        self.transitions = []
        
        
        #initialize list of all base_tasks (without identity cases)
        self.base_tasks = [base_policy(target_allocation=i.get('allocation'),
                                       name = i.get('name'), 
                                       tc = self.tc,   #could also provide tc wie input list
                                       deviation_treshold = i.get('deviation_treshold'),
                                       reward_type = self.reward_type) 
                           for i in self.base_policies_input_list ]        
        
        
        #get the first state at start of episode
        state = self._get_state()
        
        self.aux_keys = self._get_keys()
        value_dict_s = {key_:1.0 for key_ in self.aux_keys }


        #keep track how many episode env was reset to
        self.episode_count += 1
        
        return state,value_dict_s


        
    def aux_step(self):
        """
        FUNCTION TO 'MOVE' ONE TIMEPERIOD IN ENVIRONMENT

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        state : DF 
            VARIABLE REPRESENTING CURRENT STATE OF ENVIRONMENT
            element is a df with all features, .
        done : BOOL
            boolean flag indicating end of an eppoch.
        info : DICTIONARY
            currently empty - Can hold any additional info.
        aux_rewards : DICTIONARY
            Where keys are used to access given pf spec, values are float the auxiliary reward.

        """        
        aux_rewards, value_dict = self.calculate_auxiliary_rewards()

        #increase time by one
        done = self._increment_bar()
            
        #get the next state
        state = self._get_state()
        
        #if would like to have additional info back:
        info = {}        
        

        return state, done, info, aux_rewards, value_dict
    
    def _increment_bar(self):
        """
        HELPER FUNCTION TO INCREMENT CURRENT BAR (TIME IN EPOCH)
        NOTE: encapsulated in a helper function such that different cases for different environments can be overwritten/used (e.g. a dynamic increment)
        Returns
        -------
        bool
            returns flag for end of epoch (done).

        """

        # here increase time by decision interval
        self.bar += self.decision_interval
        self.eppoch_bar +=self.decision_interval
        
        #flag episode end
        if self.bar >= self.max_index - self.decision_interval-1:
            return True
        
        if self.bar >= self.end_bar - self.decision_interval-1:
            return True
        
        #end episode if have reached maximum episode duration
        if (self.bar >=  self.start_bar + self.max_episode_length):
            return True
        
        if self.eppoch_bar >= len(self.data_):
            return True
        
        return False
    
    def calculate_auxiliary_rewards(self):
        """
        HELPER FUNCTION TO CALCULATE ALL AUXILIARY REWARDS


        Returns
        -------
        ret_dict : DICTIONARY
            dictionary holding all auxiliary returns for current time period. (Keys: name of auxiliary pf, values =  rewards)

        value_dict : DICTIONARYY
            dictionary holding all auxiliary worth (total value of position) for current time period. (Keys: name of auxiliary pf, values = worth)

        """
        

        ## CALC FOR SINGLE ASSETS
        
        #e.g. Return for identity pfs (each individual asset)
        return_ = self.close_data.iloc[self.bar +self.decision_interval]/self.close_data.iloc[self.bar]

        ret_dict = dict(np.log(return_))   
        #calc wrth
        change_since_init = self.close_data.iloc[self.bar +self.decision_interval]/self.close_data.iloc[self.start_bar]
        value_dict = dict(change_since_init)
        
        #if return is change -> need to adjust for single asset case:
        if self.reward_type == 'change':
            value_start_period = self.close_data.iloc[self.bar]/self.close_data.iloc[self.start_bar]
            change_since_init = self.close_data.iloc[self.bar +self.decision_interval]/self.close_data.iloc[self.start_bar]

            change_period =  change_since_init - value_start_period
            ret_dict = dict(change_period)
            
            
        # SPECIAL CASE (MOST SIMPLE CASE) ONE ASSET ONLY
        if self.single_case:
            rel_asset = self.single_case
            
            rel_r = ret_dict.get(rel_asset)
            rel_v = value_dict.get(rel_asset)
            
            single_ret = {rel_asset:ret_dict.get(rel_asset)}
            single_v  = {rel_asset:value_dict.get(rel_asset)}
            return single_ret,single_v

            

        ## CALCULATE FOR EACH BASE TASK:    
        #for multiple portfolios have the following in a loop
        for base_p in self.base_tasks:
            len_pf = base_p.length_()
            pf_reward,pf_value  = base_p.update_state(asset_return_period= return_[0:len_pf])
            
            ret_dict[base_p.key_name] = pf_reward
            value_dict[base_p.key_name] = pf_value
        
        
        return ret_dict, value_dict

    def _get_keys(self):
        """ HELPER FUNCTION TO GET RELEVANT KEYS OF ALL AUXILIARY STRATEGIES """
        if self.single_case:
            #only one key -> of single asset
            return {self.single_case:None}.keys()
        
        aux_task_dict = {name: None for name in self.close_data.columns.str.replace('Close_','')}
        for base_p in self.base_tasks:
            aux_task_dict[base_p.key_name] = None
            
        return aux_task_dict.keys()

    def _get_env_config(self):
        """
        CREATES A DICTIONARY WITH THE RELEVANT INPUT VARIABLES ("GIVES INFO ON ENV")

        Returns
        -------
        env_config : DICT
            DICTIONARY HOLDING RELEVANT INFORMATION ABOUT ENV.

        """
        self.single_case_conf_info = self.single_case
        if self.single_case_conf_info is None:
            self.single_case_conf_info = 'all'
        
        env_config = {
            'n_lags':self.lags,
            'decision_interval':self.decision_interval,
            'max_episode_length':self.max_episode_length,
            'min_episode_length':self.min_episode_length,
            'transaction_costs':self.tc,
            'assets':self.assets,
            'end_train_date':self.end_train_date,
            'single_asset':self.single_case_conf_info,
            'reward_type':self.reward_type
            }
        return env_config    


    def _initialize_data(self):
        """
        
        FUNCTION TO (if needed) load data, clean, create features, reformat and prepare to be used
        
        -should initialize at least:
            self.close_data (a df with prices of the relevant assets (names used for close data and supplied base task specification must match)
            self.max_index  (max nr bars for given data)

        Returns
        -------
        None.
        
        
        Remark (possible ways to implement):
            
            -load all data & prepare, at reset just change index to start new epoch
                -> for steps: use index to access data (&possibly reformat)
            -preload relevant infos -> at reseet for each epoch only load relevant data & prep 
                -> for steps: use index to access data (&possibly reformat)
            -preload infos and for each step load & prep relevant data
            
            Decision: depends on size of data, loading times, features to create etc.
            
            (can use self.end_train_date to filter if want fixed end train_date)

        """
        
        # PUT YOUR OWN LOGIC HERE #
        
        
        
        
        #Simple minimal working example code
        self.close_data = self.input_data.copy()  #init data with relevant close prices
        self.dates_df = self.close_data.reset_index()[['Date']]
        self.max_index =self.max_index = len(self.dates_df) -1 
        
        
    def _get_data_for_eppoch(self):
        """
        HELPER FUNCTION
        given current start and end bar creates features df for new eppoch (will be called within reset())

        Returns
        -------
        PD DF containing features

        """
        # first get dates for current epoch
        dates_epoch = self.dates_df.iloc[self.start_bar-self.lags:self.end_bar].copy()
        
        # PUT YOUR OWN LOGIC HERE #  
        
        #Simple minimal working example code
        df = self.close_data.loc[dates_epoch.Date]
        return df
        
    def _reformat_to_state(self, df):
        """
        HELPER FUNCTION TO APPLY DIFFERENT WAYS OF FORMATTING INPUT DATA TO REPRESENT STATE 
        
        -> can format, sort, order data to have desired shape
        Parameters
        ----------
        df : DATAFRAME
            DF CONTAINING ALL RELEVANT DATA FOR CURRENT STEP.

        Returns
        -------
        df : DF
            DF FORMATTED INTO DESIRED SHAPE -> this will be state that model uses as input..

        """        
        # PUT YOUR OWN LOGIC HERE #
        
        return df

