from model import Classifier_Model
from dataset import load_dataset
from config import *
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a Convolutional neural network with configurable hyperparameters.")

    parser.add_argument("-wp", "--wandb_project", type=str, required=True,help="Project name used to track experiments in Weights & Biases dashboard.",default='cs24s031-dl-assignment')

    parser.add_argument('-cn1',"--conv1_filters",type=int,default=32,help='number of filter in conv1 layer')
    parser.add_argument('-cn2',"--conv2_filters",type=int,default=64,help='number of filter in conv2 layer')
    parser.add_argument('-cn3',"--conv3_filters",type=int,default=128,help='number of filter in conv3 layer')
    parser.add_argument('-cn4',"--conv4_filters",type=int,default=256,help='number of filter in conv4 layer')
    parser.add_argument('-cn5',"--conv5_filters",type=int,default=512,help='number of filter in conv5 layer')
    parser.add_argument('-f_org',"--filter_org",type=str,default='same',choices=['none','same','double','half'])
    parser.add_argument('-data_aug',"--data_augmentation",type=str,default='yes',choices=['yes','no'],help='wheather you want to perf data augmentation or not')
    parser.add_argument('-bn',"--batch_norm",type=str,default='yes',choices=['yes','no'],help='perfor batch normalization')
    parser.add_argument('-do',"--dropout",type=float,default=0.2,choices=[0.2,0.3,0.5,0.4,0.7,0.6],help='apply dropout')
    parser.add_argument('-ls',"--dense_layer_neurons",type=int,default=2048,help='number of neurons in the dense network')

    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='rajshekharrakshit')    
    parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,help="Batch size used to train the neural network.")

    parser.add_argument("-o", "--optimizer", type=str,default='nadam' ,choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help="Optimizer to be used. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'].")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,help="Learning rate used to optimize model parameters.")

    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default='Xavier', choices=["random", "Xavier"], help="Weight initialization method. Choices: ['random', 'Xavier'].")

    parser.add_argument("-a", "--activation", type=str,default='relu', choices=["sigmoid",'relu','tanh','mish','silu','selu'], help="Activation function to be used. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU'].")
    parser.add_argument("--output_shape",type=int,default=10,help="Number of neuron in output layer.")
    parser.add_argument('-pc',"--patience_counter",type=int,default=10,help="Patience Counter for Early stopping")   

    args = parser.parse_args()
    
    weight_decay = args.weight_decay
    weight_init = args.weight_init
    activation = args.activation
    hidden_neurons = args.dense_layer_neurons
    lr = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    out_shape = args.output_shape
    dropout = args.dropout
    patience_counter = args.patience_counter

    optimizer_kwargs = {}
    optimizer_kwargs['weight_decay'] = weight_decay
    optimizer_kwargs['weght_init'] = weight_init
    optimizer_kwargs['activation'] = activation
    optimizer_kwargs['hidden_neurons'] = hidden_neurons
    optimizer_kwargs['lr'] = lr
    optimizer_kwargs['weight_decay'] = weight_decay
    optimizer_kwargs['batch_size'] = batch_size
    optimizer_kwargs['out_shape'] = out_shape
    optimizer_kwargs['dropout'] = dropout
    optimizer_kwargs['data_aug'] = args.data_augmentation
    optimizer_kwargs['batch_norm'] = args.batch_norm
    optimizer_kwargs['patience_counter'] = patience_counter


    print(optimizer_kwargs)

    train_dataset,val_dataset,test_dataset = load_dataset('../inaturalist_12K/')

    cn1_filters = args.conv1_filters
    cn2_filters = args.conv2_filters
    cn3_filters = args.conv3_filters
    cn4_filters = args.conv4_filters
    cn5_filters = args.conv5_filters

    model = Classifier_Model(out_classes=10,cn1_filters=cn1_filters,cn2_filters=cn2_filters,cn3_filters=cn3_filters,output_shape=out_shape,
                             cn4_filters=cn4_filters,cn5_filters=cn5_filters,cn1_kernel_size=3,cn2_kernel_size=3,cn3_kernel_size=3,cn4_kernel_size=3,
                             cn5_kernel_size=3,n_dense_output_neuron=hidden_neurons,filter_organisation='same',batch_normalization='yes',dropout=dropout)
    
    model.train_network(train_data=train_dataset,val_data=val_dataset,test_data=test_dataset,
                        batch_size=batch_size,lr=lr,weight_decay=weight_decay,early_stopping_patience=patience_counter)
    
