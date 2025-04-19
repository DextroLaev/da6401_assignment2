"""
Hyperparameter Sweep Training Script for CNN on iNaturalist 12K Dataset

This script performs a hyperparameter sweep to train a configurable convolutional neural network (CNN)
on the iNaturalist 12K dataset using Weights & Biases (wandb). The goal is to optimize model performance
by tuning various hyperparameters using Bayesian search.

Key Features:
- Uses wandb for hyperparameter sweep management and experiment tracking.
- CNN architecture is defined in the `Classifier_Model` class.
- The model is trained on train and validation splits from iNaturalist 12K.
- Supports a wide range of hyperparameters including activation functions, dropout, filter organization, and more.
- Automatically assigns a descriptive run name based on key hyperparameters.
- Clears GPU memory after each run to prevent memory leaks.

Hyperparameters (sweep_config):
- hidden_neurons: Number of neurons in the dense (fully connected) layer.
- activation_function: Activation function to be used in conv/dense layers.
- batch_size: Training batch size.
- learning_rate: Learning rate for optimizer (e.g., Adam).
- weight_decay: Weight decay (L2 regularization).
- data_aug: Whether to apply data augmentation to training data.
- batch_normalization: Whether to include BatchNorm layers in the model.
- num_filters: Number of filters in the first conv layer.
- dropout: Dropout rate applied after convolution blocks.
- filter_organisation: How filters evolve across layers ('same', 'double', 'half').

Notes:
- WandB login is required before running (wandb.login()).
- Sweep is executed with 100 runs using the Bayesian optimization strategy.
- Make sure the dataset path ('../inaturalist_12K/') is correct.
"""


from dataset import load_dataset
from model import Classifier_Model
from config import *
import wandb
import os
import torch
import gc

def train():

    wandb.login()
    var1 = wandb.init(project='dl-assignment2')

    
    config = var1.config

    hidden_neurons = config.hidden_neurons
    activation_function = config.activation_function
    weight_decay = config.weight_decay
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    data_aug = config.data_aug
    batch_norm = config.batch_normalization
    dropout = config.dropout
    filter_org = config.filter_organisation
    num_filters = config.num_filters


    train_data,val_data,test_data = load_dataset('../inaturalist_12K/',input_shape=(256,256),data_aug=data_aug,batch_size=batch_size)
    

    run_name = f"filter_{filter_org}_bs_{config.batch_size}_ac_{config.activation_function}"
    
    wandb.run.name = run_name
    wandb.run.save()
  
    print(f"Starting training with run name: {run_name}")
    model = Classifier_Model(out_classes=10,n_dense_output_neuron=hidden_neurons,activation=activation_function,
                             filter_organisation=filter_org,num_filters=num_filters,
                             batch_normalization=batch_norm,dropout=dropout)
    
    model.train_network(train_data=train_data,
                        val_data=val_data,                        
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        epochs=20,log=True)

    model.to('cpu')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    sweep_config = {
		'name': 'inaturalist12K-exp(bayes-select)',
		'method': 'bayes',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
		    'hidden_neurons':{'values':[1024,2048,4192]},		    
		    'activation_function': {'values': ['tanh', 'relu','silu','selu','mish','leaky_relu','gelu']},
		    'batch_size': {'values': [32,64]},
		    'learning_rate': {'values': [1e-4,1e-5,2e-4]},
		    'weight_decay': {'values': [0, 0.0005,0.00005]},
            'data_aug':{'values':[True,False]},
            'batch_normalization':{'values':[True,False]},
            'num_filters':{'values':[32,64,128]},
            'dropout':{'values':[0.2,0.3]},
            'filter_organisation':{'values':['same','double','half']}
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment2')
    wandb.agent(sweep_id,train,count=100)
    wandb.finish()
