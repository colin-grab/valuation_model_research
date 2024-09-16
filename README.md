# valuation_model_research
Example implementation of models used for my research <br>
(see pre-print of research on https://arxiv.org/abs/2405.11686 )

    
#### Note:
Code available has been simplified with some minimal working example code - need to put your own calculations/logic for:
    
    - Data handling in environment 
    (inside environments.py, functions: _initialize_data(), _get_data_for_eppoch(), _reformat_to_state())
    
    - Neural net structure


### Setup via Anaconda:

    conda create -n YOUR_ENV_NAME python=3.9
    conda activate YOUR_ENV_NAME 
    conda install pyopengl pandas numpy tqdm keras tensorflow pytables
