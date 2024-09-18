# -*- coding: utf-8 -*-
"""

EXAMPLE NEURAL NET CLASS FOR VALUATION MODEL

@author: Colin Grab

"""



from imports import *

class cdg_example_nn(keras.Model):
    """ Neural Net for CG or CDG model 
        i.e. multiple value functions or distributional value functions according to auxiliary tasks, 
             allows for multiple gammas
             
             aux_keys : Name of the auxiliary tasks, any iterable
             aux_dist : Boolean Flag if distributional variant 
             n_gammas : Number of gammas to be used 
             n_atoms  : Number of bins that are fitted for distributional value function
             
    Call: cg_and_cdg_network(shape_input,aux_keys = {}.keys(), aux_dist = False, gammas = [0.99],n_atoms = 51)
    """
    def __init__(self, shape_input,aux_keys = {}.keys(), aux_dist = False, n_gammas =1 ,n_atoms = 51):
        super(cdg_example_nn, self).__init__()
        
        self.aux_keys = aux_keys
        self.aux_dist = aux_dist
        self.n_atoms =n_atoms
        
        self.n_gammas = n_gammas

        self.shape_input = shape_input
         
        
        #some example layers (replace with your own layers, logic)
        self.fc1 = Dense(50, activation='gelu',kernel_initializer="lecun_normal")
        self.drop_out_1 = layers.Dropout(0.2, name ='drop_out_1')
        self.flat = Flatten()        
        self.emb_layer = Dense(50, activation=None,  name = 'X_embedding_layer')
        
        
        # Creating layers to output the estimate for auxiliary value or auxiliary dist function (here use own head for each task (could also directly use #units layer))
        
        if self.aux_dist:
            self.aux_layers = [Dense(self.n_gammas * self.n_atoms,activation = None, kernel_initializer="he_uniform", 
                                     name = 'aux_logits_'+str(j)) for j in self.aux_keys]
            self.aux_softmax = [Softmax(name = 'softmax_auxg_'+str(aux_k)) for aux_k in self.aux_keys]
        else:
            self.aux_layers = [Dense(self.n_gammas,activation = None, kernel_initializer="he_uniform", 
                                     name = 'aux_v_'+str(j))
                               for j in self.aux_keys]
            
                
    def call(self, state):
        

        x=state
        x = self.fc1(x)
        x = self.drop_out_1(x)
        x = self.flat(x)
        
        # ----------------
        #to increase network, can add different layers in init and add layers here

        # PUT YOUR OWN LOGIC HERE
        
        #-----------------
        embedding_X = self.emb_layer(x)
        
        #CDG output
        if self.aux_dist:
            aux_multi = [aux_softmax(tf.reshape(layer(embedding_X),[-1,self.n_gammas,self.n_atoms])) 
                         for layer,aux_softmax in zip(self.aux_layers,self.aux_softmax)]
            return embedding_X, aux_multi
    
        #else CG output
        aux_multi = [layer(embedding_X) for layer in self.aux_layers]
        
        return embedding_X, aux_multi
    
    def model(self):
        input1 = tf.keras.layers.Input(shape = self.shape_input)
        return keras.Model(inputs = [input1], outputs = self.call(input1))
