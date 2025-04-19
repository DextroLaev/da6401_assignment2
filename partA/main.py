"""
    Train Configurable CNN on iNaturalist 12K Dataset.

    This script is used to train a custom convolutional neural network (CNN) model for image classification
    on the iNaturalist 12K dataset using configurable hyperparameters. The CNN consists of 5 convolutional blocks
    followed by a dense layer and an output layer. Each block contains a Conv2D -> Activation -> MaxPool2D structure.

    Key Features:
    - Adjustable input image size (square)
    - Configurable number and organization of filters in convolutional layers
    - Flexible activation function choices
    - Optional batch normalization and dropout
    - Dense layer with configurable number of neurons
    - Optional data augmentation
    - Support for logging and visualizing experiments with Weights & Biases (wandb)
    - Early stopping and optional model checkpoint saving

    Arguments:
    - --wandb_project (-wp): WandB project name (required if logging)
    - --wandb_entity (-we): WandB entity/user/team name (required if logging)
    - --input_shape (-input_s): Size of square input image (e.g., 256)
    - --filter_org (-f_org): Filter organization pattern ('same', 'double', 'half')
    - --num_filters (-n_filters): Number of filters in the first conv layer (e.g., 64)
    - --data_augmentation (-data_aug): Enable/disable data augmentation (True/False)
    - --batch_norm (-bn): Enable/disable batch normalization (True/False)
    - --dropout (-do): Dropout rate (e.g., 0.3)
    - --dense_layer_neurons (-ls): Number of neurons in the dense layer
    - --epochs (-e): Number of training epochs
    - --batch_size (-b): Training batch size
    - --learning_rate (-lr): Learning rate for optimizer
    - --weight_decay (-w_d): Weight decay (L2 regularization)
    - --activation (-a): Activation function ('relu', 'tanh', 'mish', etc.)
    - --output_classes (--output_classes): Number of output classes (default: 10)
    - --patience_counter (-pc): Early stopping patience
    - --save_model (--save_model): Save model checkpoint (True/False)
    - --log_wandb (-logw): Enable logging to wandb (True/False)
"""

from model import Classifier_Model
from dataset import load_dataset
from config import *
import argparse
import wandb

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Train a Convolutional neural network with configurable hyperparameters.")

    parser.add_argument("-wp", "--wandb_project", type=str, required=True,help="Project name used to track experiments in Weights & Biases dashboard.",default='cs24s031-dl-assignment2')

    
    parser.add_argument('-input_s',"--input_shape",type=int,default=256,choices=[256,128,400,224],help='give the input shape of the image ( square matrix ), just give one dimension only')
    parser.add_argument('-f_org',"--filter_org",type=str,default='double',choices=['same','double','half'])
    parser.add_argument('-n_filters',"--num_filters",type=int,default=64,choices=[32,64,128])
    parser.add_argument('-data_aug',"--data_augmentation",type=bool,default=False,choices=[True,False],help='wheather you want to perf data augmentation or not')
    parser.add_argument('-bn',"--batch_norm",type=bool,default=True,choices=[True,False],help='perfor batch normalization')
    parser.add_argument('-do',"--dropout",type=float,default=0.3,choices=[0.2,0.3,0.5,0.4,0.7,0.6],help='apply dropout')


    parser.add_argument('-ls',"--dense_layer_neurons",type=int,default=4192,help='number of neurons in the dense network')
    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='rajshekharrakshit')    
    parser.add_argument("-e", "--epochs", type=int, default=25,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,help="Batch size used to train the neural network.")
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,help="Learning rate used to optimize model parameters.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.00005,help="Weight decay used by optimizers.")

    parser.add_argument("-a", "--activation", type=str,default='selu', choices=['relu','tanh','mish','silu','selu','gelu','leaky_relu'], help="Activation function to be used.")
    parser.add_argument("--output_classes",type=int,default=10,help="Number of neuron in output layer.")
    parser.add_argument('-pc',"--patience_counter",type=int,default=10,help="Patience Counter for Early stopping")
    parser.add_argument('--save_model',type=bool,default=False,choices=[True,False],help="If you want to save the trained model, set it to True")
    parser.add_argument('-logw',"--log_wandb",type=bool,default=False,choices=[True,False],help="If you want to log the performance of the model, set it to True")

    args = parser.parse_args()
    
    weight_decay = args.weight_decay
    activation = args.activation
    hidden_neurons = args.dense_layer_neurons
    lr = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    out_classes = args.output_classes
    dropout = args.dropout
    patience_counter = args.patience_counter
    save_model = args.save_model
    log_wandb = args.log_wandb
    num_filters = args.num_filters
    filter_org = args.filter_org
    batch_norm = args.batch_norm
    input_shape = (args.input_shape,args.input_shape)
    data_aug = args.data_augmentation
    epochs = args.epochs

    if log_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project,entity=args.wandb_entity)

    train_dataset,val_dataset,test_dataset = load_dataset('../inaturalist_12K/',data_aug=data_aug,batch_size=batch_size,input_shape=input_shape)

    model = Classifier_Model(out_classes=out_classes,n_dense_output_neuron=hidden_neurons,
                             num_filters=num_filters,activation=activation,filter_organisation=filter_org,
                             batch_normalization=batch_norm,dropout=dropout,input_shape=input_shape
                             )
    
    model.train_network(train_data=train_dataset,val_data=val_dataset,lr=lr,epochs=epochs,
                        weight_decay=weight_decay,early_stopping_patience=patience_counter,log_wandb=log_wandb,save_model=save_model)
    
