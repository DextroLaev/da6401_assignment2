from dataset import load_dataset
from model import Classifier_Model
import matplotlib.pyplot as plt
from config import *
import wandb
import os
import torch


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
    filter_size = config.filter_size


    train_data,val_data,test_data = load_dataset('./inaturalist_12K/',input_shape=(256,256),data_aug=data_aug)
    

    run_name = f"filter_{filter_org}_bs_{config.batch_size}_ac_{config.activation_function}"
    
    wandb.run.name = run_name
    wandb.run.save()
  
    print(f"Starting training with run name: {run_name}")
    model = Classifier_Model(out_classes=10,n_dense_output_neuron=hidden_neurons,activation=activation_function,
                             filter_organisation=filter_org,filter_size=filter_size,
                             batch_normalization=batch_norm,dropout=dropout)
    
    model.train_network(train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        epochs=20,batch_size=batch_size,log=True)
    del model
    del train_data
    del val_data
    del test_data
    torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    sweep_config = {
		'name': 'inaturalist12K-exp(bayes-select)',
		'method': 'bayes',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
		    'hidden_neurons':{'values':[1024,2048,4192]},		    
		    'activation_function': {'values': ['sigmoid', 'tanh', 'relu','silu','selu','mish','leaky_relu']},
		    'batch_size': {'values': [16,32,64]},
		    'learning_rate': {'values': [1e-4,1e-5,2e-5]},		    		    
		    'weight_decay': {'values': [0, 0.0005,0.00005]},
            'data_aug':{'values':['yes','no']},
            'batch_normalization':{'values':['yes','no']},
            'filter_size':{'values':[32,64,128,256,512]},
            'dropout':{'values':[0.2,0.3]},
            'filter_organisation':{'values':['same','double','half']}
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment2')
    wandb.agent(sweep_id,train,count=60)
    wandb.finish()
