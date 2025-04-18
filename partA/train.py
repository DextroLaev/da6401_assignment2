import torch
from dataset import load_dataset
from model import Classifier_Model
import matplotlib.pyplot as plt
from config import *
import wandb

def train():
    """
    Initializes Weights & Biases logging, loads the dataset, defines the model,
    and starts the training process.

    The training uses a custom CNN model (Classifier_Model) on the iNaturalist 12K dataset.
    The dataset is split into training, validation, and test sets. The model is configured
    with specific hyperparameters such as architecture details, learning rate, dropout, 
    batch normalization, etc. Training is then performed for a fixed number of epochs.

    This function also initializes W&B logging but sets `log=False` for training, so metrics
    will not be uploaded unless changed.
    Notes
    -----
    - The model is trained on GPU if CUDA is available.
    - Dataset should follow the expected structure with 'train' and 'val' directories.
    - Weights & Biases must be logged into prior to running.
    - `config.py` should define the `DEVICE` variable.

    Returns
    -------
    None
    """
    wandb.login()
    var1 = wandb.init(project='dl-assignment2')

    train_data,val_data,test_data = load_dataset('./inaturalist_12K/')
    
    model = Classifier_Model(out_classes=10,n_dense_output_neuron=2048,activation='relu',
                             filter_organisation='double',num_filters=32,
                             batch_normalization='yes',dropout=0.5)
    
    model.train_network(train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        lr=1e-5,
                        weight_decay=0.0,
                        epochs=20,log=False)


if __name__ == '__main__':
    train()