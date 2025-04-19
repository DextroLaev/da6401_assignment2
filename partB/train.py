"""
Main training script for the model on the iNaturalist 12K dataset.

This script initializes the training process for a pretrained ResNet50 model using a specified 
training method (last layer, partial, or gradual fine-tuning). The dataset is loaded, and the model 
is trained for a specified number of epochs. The training process can log results to Weights & Biases 
and save the trained model.

Parameters:
-----------
- The model is initialized using hyperparameters defined in the `config.py` file, such as:
    - `BATCH_SIZE`: The batch size used for training.
    - `DATA_AUG`: Flag to enable or disable data augmentation.
    - `TRAIN_METHOD`: Fine-tuning strategy (`'last_layer'`, `'partial'`, or `'gradual'`).
    - `TRAIN_LAST_K_LAYERS`: Number of last layers to fine-tune (used in 'partial' method).
    - `TRAIN_NEW_FC_STEP`: Step size for unfreezing layers in the 'gradual' fine-tuning method.
    - `OUT_CLASSES`: The number of output classes for classification.
    - `LEARNING_RATE`: The learning rate used for the optimizer.
    - `EPOCHS`: The number of epochs to train the model.
    - `LOG_WANDB`: Flag to log results to Weights & Biases.
    - `SAVE_MODEL`: Flag to save the trained model.
    - `EARLY_STOPPING_PATIENCE`: The patience parameter for early stopping.
    - `WEIGHT_DECAY`: The weight decay for regularization.

Function Flow:
--------------
1. Logs into Weights & Biases if `LOG_WANDB` is set to True.
2. Initializes a Weights & Biases run for experiment tracking.
3. Loads the iNaturalist 12K dataset with specified batch size and data augmentation settings.
4. Initializes a `PretrainedModel` with the provided training method and hyperparameters.
5. Starts training the model using the `train_network` method, which logs the training process 
   and saves the best model if `SAVE_MODEL` is set to True.
6. Optionally applies early stopping if `EARLY_STOPPING_PATIENCE` is set.

Returns:
--------
None
"""


from dataset import load_dataset
from model import PretrainedModel
from config import *
import wandb

if __name__ == '__main__':

    if LOG_WANDB:
        wandb.login()
        wandb.init(project='dl-assignment2')

    train_data,val_data,test_data = load_dataset('../inaturalist_12K',batch_size=BATCH_SIZE,data_aug=DATA_AUG)

    model = PretrainedModel(train_method=TRAIN_METHOD,train_last_k_layers=TRAIN_LAST_K_LAYERS,train_new_fc_step=TRAIN_NEW_FC_STEP,out_classes=OUT_CLASSES)

    model.train_network(train_data,val_data,lr=LEARNING_RATE,log_wandb=LOG_WANDB,epochs=EPOCHS,
                        save_model=SAVE_MODEL,early_stopping_patience=EARLY_STOPPING_PATIENCE,weight_decay=WEIGHT_DECAY)