"""
Train a configurable ResNet50-based classifier on the iNaturalist 12K dataset using transfer learning.

This script provides a command-line interface (CLI) to flexibly fine-tune a pretrained ResNet50 model 
using different training strategies and hyperparameters. It supports training only the last layer, 
partial unfreezing of layers, or gradual unfreezing of deeper layers. The model is trained using the 
PretrainedModel class and the dataset is loaded using the `load_dataset` utility.

Key Features:
-------------
- Supports training the full network, only the final layers, or gradual unfreezing.
- Configurable input image resolution, dropout, batch normalization, and dense layer size.
- Optional Weights & Biases (wandb) logging and model checkpoint saving.
- Early stopping support with customizable patience.

CLI Arguments:
--------------
- --wandb_project, --wandb_entity: Define the wandb tracking details.
- --input_shape: Image size (e.g., 224 or 256 for ResNet50 input).
- --data_augmentation: Enable/disable data augmentation.
- --batch_norm: Enable/disable batch normalization.
- --dropout: Dropout rate (e.g., 0.2, 0.5).
- --train_method: Choose training strategy: 'last_layer', 'partial', or 'gradual'.
- --train_last_k_layers: Number of layers to unfreeze (only for 'partial' training).
- --train_new_fc_step: Step size for unfreezing layers during gradual training.
- --dense_layer_neurons: Number of neurons in the dense fully-connected layer.
- --epochs: Total number of training epochs.
- --batch_size: Batch size used for training.
- --learning_rate: Optimizer learning rate.
- --weight_decay: Weight decay (L2 regularization).
- --output_classes: Number of output classes in the dataset.
- --patience_counter: Early stopping patience.
- --save_model: Whether to save the model after training.
- --log_wandb: Whether to log training metrics to wandb.

Returns:
--------
None

Example Usage:
--------------
python train_resnet.py --wandb_project my_project --wandb_entity my_entity --train_method partial --train_last_k_layers 3 --log_wandb True
"""


from model import PretrainedModel
from dataset import load_dataset
from config import *
import argparse
import wandb


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a Convolutional neural network with configurable hyperparameters.")

    parser.add_argument("-wp", "--wandb_project", type=str, required=True,help="Project name used to track experiments in Weights & Biases dashboard.",default='cs24s031-dl-assignment')
    
    
    parser.add_argument('-input_s',"--input_shape",type=int,default=256,choices=[256,128,400,224],help='give the input shape of the image ( square matrix ), just give one dimension only')
    
    parser.add_argument('-data_aug',"--data_augmentation",type=bool,default=True,choices=[True,False],help='wheather you want to perf data augmentation or not')
    parser.add_argument('-t_method',"--train_method",type=str,default='last_layer',choices=['last_layer','gradual','partial'],help='Which layers you want to fine-tune the ResNet50')
    parser.add_argument('-t_last_k_layers',"--train_last_k_layers",type=int,default=2,choices=[1,2,3,4,5,6,7],help='Uptill which layer (from last) you want to fine-tune the model. It is needed only when train_method is partila')
    parser.add_argument('-t_new_fc_step',"--train_new_fc_step",type=int,default=3,help='This is used when train_model is gradual, this value is used to unfreeze each prev layer when epochs % train_new_fc_step is 0')

    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='rajshekharrakshit')    
    parser.add_argument("-e", "--epochs", type=int, default=20,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,help="Batch size used to train the neural network.")
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,help="Learning rate used to optimize model parameters.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,help="Weight decay used by optimizers.")

    parser.add_argument("--output_classes",type=int,default=10,help="Number of neuron in output layer.")
    parser.add_argument('-pc',"--patience_counter",type=int,default=10,help="Patience Counter for Early stopping")
    parser.add_argument('--save_model',type=bool,default=False,choices=[True,False],help="If you want to save the trained model, set it to True")
    parser.add_argument('-logw',"--log_wandb",type=bool,default=False,choices=[True,False],help="If you want to log the performance of the model, set it to True")

    args = parser.parse_args()

    lr = args.learning_rate
    batch_size = args.batch_size
    out_classes = args.output_classes

    patience_counter = args.patience_counter
    save_model = args.save_model
    log_wandb = args.log_wandb
    input_shape = (args.input_shape,args.input_shape)
    data_aug = args.data_augmentation
    epochs = args.epochs
    train_method = args.train_method
    train_last_k_layers = args.train_last_k_layers
    train_new_fc_step = args.train_new_fc_step

    if log_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project,entity=args.wandb_entity)

    train_dataset,val_dataset,test_dataset = load_dataset('../inaturalist_12K/',data_aug=data_aug,batch_size=batch_size,input_shape=input_shape)

    model = PretrainedModel(out_classes=out_classes,train_method=train_method,
                            train_last_k_layers=train_last_k_layers,
                            train_new_fc_step=train_new_fc_step,input_shape=input_shape
                            )
    
    model.train_network(train_data=train_dataset,val_data=val_dataset,lr=lr,epochs=epochs,early_stopping_patience=patience_counter,log_wandb=log_wandb,save_model=save_model)
    
