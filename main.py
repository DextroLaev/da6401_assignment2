from train import train_network
from dataset import DataLoader
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
    parser.add_argument('-f_org',"--filter_org",type=str,default='same',choices=['same','double','half'])
    parser.add_argument('-data_aug',"--data_augmentation",type='str',default='yes',choices=['yes','no'],help='wheather you want to perf data augmentation or not')
    parser.add_argument('-bn',"--batch_norm",type='str',default='yes',choices=['yes','no'],help='perfor batch normalization')
    parser.add_argument('-do',"--dropout",type=float,default=0.2,choices=[0.2,0.3],help='apply dropout')
    parser.add_argument('-ls',"--dense_layer_neurons",type=int,default=2048,help='number of neurons in the dense network')

    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='rajshekharrakshit')    
    parser.add_argument("-e", "--epochs", type=int, default=100,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,help="Batch size used to train the neural network.")

    parser.add_argument("-o", "--optimizer", type=str,default='nadam' ,choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help="Optimizer to be used. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'].")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,help="Learning rate used to optimize model parameters.")

    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default='Xavier', choices=["random", "Xavier"], help="Weight initialization method. Choices: ['random', 'Xavier'].")

    parser.add_argument("-a", "--activation", type=str,default='ReLU', choices=["identity","sigmoid", "tanh", "ReLU"], help="Activation function to be used. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU'].")
    parser.add_argument("--output_shape",type=int,default=10,help="Number of neuron in output layer.")    

    args = parser.parse_args()
    
    weight_decay = args.weight_decay
    weight_init = args.weight_init
    num_neurons_dense_layer =  args.dense_layer_neurons    
    activation = args.activation
    optimizer_kwargs = {}
    if args.optimizer in ["momentum", "nag"]:
        optimizer_kwargs["momentum"] = args.momentum
    elif args.optimizer == "rmsprop":
        optimizer_kwargs["beta"] = args.beta
        optimizer_kwargs["eps"] = args.epsilon
    elif args.optimizer in ["adam", "nadam"]:
        optimizer_kwargs["beta1"] = args.beta1
        optimizer_kwargs["beta2"] = args.beta2
        optimizer_kwargs["eps"] = args.epsilon


    (train_data,train_label),(test_data,test_label),(val_data,val_label) = Dataset(dataset).load_data()
    input_shape = train_data.shape[1]
    output_shape = args.output_shape
    wandb.init(project=args.wandb_project,name=args.wandb_entity)
    
    nn = Neural_Net(input_shape = input_shape,output_shape = output_shape,
    	number_of_hidden_layers=num_hidden_layers, hidden_neurons_per_layer=num_neurons_each_layer,
    	activation_name=activation,type_of_init=weight_init,L2reg_const=weight_decay)
    nn.train(
        optimizer=args.optimizer,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss_type=args.loss,
        train_data=train_data,
        train_label=train_label,
        val_data=val_data,
        val_label=val_label,
        test_data = test_data,
        test_label = test_label,
        batch_size=args.batch_size,
        **optimizer_kwargs
    )
    test_loss, test_acc = nn.test_accuracy_loss(test_data,test_label)
    print('Test loss:- ',test_loss)
    print('Test Acc:- ',test_acc)
    wandb.finish()

