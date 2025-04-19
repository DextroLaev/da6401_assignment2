import torch
from dataset import load_dataset
from model import Classifier_Model
import matplotlib.pyplot as plt
from config import *
import wandb

def train():
    """
    Trains a custom CNN model (Classifier_Model) on the iNaturalist 12K dataset.

    This function performs the following steps:
        - Logs into Weights & Biases (wandb) and initializes a logging run.
        - Loads the iNaturalist dataset with training, validation, and test splits using predefined settings.
        - Initializes the Classifier_Model with architecture and training hyperparameters from config.
        - Trains the model using the specified optimizer, loss, and learning rate schedule.
        - Optionally logs metrics to wandb and saves the best-performing model to disk.
        - Supports early stopping based on validation performance.

    Configuration:
        - All training and model hyperparameters are imported from `config.py`.
        - Logging, dropout, batch normalization, and data augmentation settings are customizable.

    Notes:
        - The model is trained on GPU if available, otherwise defaults to CPU.
        - The dataset path must contain 'train' and 'val' subdirectories.
        - You must be logged into Weights & Biases (wandb) for logging to function properly.

    Returns:
        None
    """
    wandb.login()
    var1 = wandb.init(project='dl-assignment2')

    train_data,val_data,test_data = load_dataset('../inaturalist_12K/',data_aug=DATA_AUG,batch_size=BATCH_SIZE)
    
    model = Classifier_Model(out_classes=OUT_CLASSES,n_dense_output_neuron=DENSE_NEURONS,activation=ACTIVATION,
                             filter_organisation=FILTER_ORGANISATION,num_filters=NUM_FILTERS,
                             batch_normalization=BATCH_NORMALIZATION,dropout=DROPOUT)
    
    model.train_network(train_data=train_data,
                        val_data=val_data,
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        epochs=EPOCHS,log_wandb=LOG_WANDB,save_model=SAVE_MODEL,early_stopping_patience=EARLY_STOPPING_PATIENCE)


if __name__ == '__main__':
    train()
