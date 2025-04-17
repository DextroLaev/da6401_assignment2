import torch
from dataset import load_dataset
from model import Classifier_Model
import matplotlib.pyplot as plt
from config import *
import wandb

def train():
    wandb.login()
    var1 = wandb.init(project='dl-assignment2')

    train_data,val_data,test_data = load_dataset('./inaturalist_12K/')
    
    model = Classifier_Model(out_classes=10,n_dense_output_neuron=2048,activation='relu',cn1_filters=32,cn2_filters=64,
                             cn3_filters=128,cn4_filters=256,cn5_filters=512,
                             filter_organisation='double',filter_size=32,
                             batch_normalization='yes',dropout=0.5)
    
    model.train_network(train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        lr=1e-5,
                        weight_decay=0.0,
                        epochs=20,batch_size=32,log=False)


if __name__ == '__main__':
    train()