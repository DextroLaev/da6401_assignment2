"""
PretrainedModel Class for Fine-Tuning ResNet50 on Custom Dataset

This class is designed to fine-tune a pretrained ResNet50 model on a custom dataset. It provides multiple
training strategies, including training only the last layer, partial training of the network, and gradual
unfreezing of layers. The model uses a standard ResNet50 architecture with a custom fully connected layer 
added to match the number of output classes.

Attributes:
-----------
- model (torch.nn.Module): Pretrained ResNet50 model.
- output_classes (int): Number of output classes for the model's final layer.
- loss (torch.nn.CrossEntropyLoss): Loss function used for training.
- last_layer (torch.nn.Linear): Fully connected layer to match output_classes.
- train_method (str): Method for training the network ('last_layer', 'partial', 'gradual').
- train_last_k_layers (int): Number of layers to train when using 'partial' training method.
- train_new_fc_step (int): Step size for unfreezing layers during 'gradual' training method.
- gradual (bool): Flag indicating if gradual unfreezing is used.
- input_shape (tuple): Shape of the input image.
- resnet50_layer_names (list): List of names of layers in the ResNet50 architecture.
- layers_getting_trained (list): List of layers being trained.

Methods:
--------
- __init__: Initializes the pretrained ResNet50 model and configures it based on the chosen training method.
- model_init: Configures the layers to be trained based on the specified training method.
- set_prev_layer_train: Sets the previous layers to be trained in gradual unfreezing mode.
- forward: Forward pass through the model.
- train_network: Trains the model on the provided training and validation data using the specified parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config import *
from torchvision.models import resnet50, ResNet50_Weights

class PretrainedModel(nn.Module):
    def __init__(self,out_classes=10,input_shape=(256,256),train_method='last_layer',train_last_k_layers=2,train_new_fc_step=0):
        """
        Initializes the PretrainedModel by loading the ResNet50 model and configuring it based on the 
        specified training method. Adds a custom final fully connected layer for the specified number 
        of output classes.

        Parameters:
        -----------
        out_classes (int): The number of output classes (default: 10).
        input_shape (tuple): The shape of the input images (default: (256, 256)).
        train_method (str): Method for training the model ('last_layer', 'partial', 'gradual') (default: 'last_layer').
        train_last_k_layers (int): Number of layers to unfreeze when using 'partial' training (default: 2).
        train_new_fc_step (int): Step size for unfreezing layers during 'gradual' training (default: 0).
        """
        super(PretrainedModel,self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.output_classes = out_classes
        self.loss = nn.CrossEntropyLoss()
        self.last_layer = nn.Linear(self.model.fc.in_features,self.output_classes)        
        self.train_method = train_method
        self.train_last_k_layers = train_last_k_layers 
        self.train_new_fc_step = train_new_fc_step
        self.gradual = False
        self.input_shape=input_shape
        self.resnet50_layer_names = ['conv1','bn1','layer1','layer2','layer3','layer4','fc']
        self.layers_getting_trained = self.model_init()
        
        self.model.to(DEVICE)

    def model_init(self):
        """
        Initializes the layers of the model based on the chosen training method.
        
        Returns:
        --------
        list: List of layers that will have their weights updated during training.
        """
        if self.train_method == 'last_layer':
            for params in self.model.parameters():
                params.requires_grad = False
            
            self.model.fc = self.last_layer
            setattr(self.model,'new_fc',self.model.fc)
            return ['new_fc']
        
        elif self.train_method == 'partial':
            for params in self.model.parameters():
                params.requires_grad = False
            
            train_layers = self.resnet50_layer_names[-self.train_last_k_layers:]
            for name,param in self.model.named_parameters():
                if name in train_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.model.fc = self.last_layer
            setattr(self.model,'new_fc',self.model.fc)
            self.resnet50_layer_names.append('new_fc')
            return train_layers+['new_fc']
        
        elif self.train_method == 'gradual':
            self.gradual = True
            self.model.fc = self.last_layer
            setattr(self.model,'new_fc',self.model.fc)
            self.resnet50_layer_names.append('new_fc')
            return ['new_fc']
    
    def set_prev_layer_train(self):
        """
        Gradually unfreezes previous layers during training when using the 'gradual' training method.
        This method adds layers to the list of layers being trained.
        """
        
        recently_set_layer_train_index = self.resnet50_layer_names.index(self.layers_getting_trained[0])
        if recently_set_layer_train_index > 0 and recently_set_layer_train_index < len(self.resnet50_layer_names):
            self.layers_getting_trained.insert(0,self.resnet50_layer_names[recently_set_layer_train_index-1])

            for name, param in self.model.named_parameters():
                for layer in self.layers_getting_trained:
                    if name.startswith(layer):
                        param.requires_grad = True
                        break
                else:
                    param.requires_grad = False
    
    def forward(self,x):
        """
        Forward pass through the model.
        Parameters:
        -----------
        x (torch.Tensor): Input tensor to the model.
        Returns:
        --------
        torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)

    def train_network(self,train_data,val_data,epochs,lr=1e-3,log_wandb=False,save_model=False,model_save_path='./models/model.pth',early_stopping_patience=5,weight_decay=0.0):        
        """
        Trains the model using the specified training and validation data, optimizer, and parameters.

        Parameters:
        -----------
        train_data (DataLoader): Training data loader.
        val_data (DataLoader): Validation data loader.
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer (default: 1e-3).
        log_wandb (bool): Whether to log metrics to Weights & Biases (default: False).
        save_model (bool): Whether to save the model after training (default: False).
        model_save_path (str): Path where the model will be saved (default: './models/model.pth').
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping (default: 5).
        weight_decay (float): Weight decay for the optimizer (default: 0.0).
        """
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        for ep in range(epochs):
            self.model.train()
            train_loss,acc,total = 0.0,0,0
            for img,label in train_data:
                img,label = img.to(DEVICE),label.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(img)
                loss_val = loss_fn(output,label)
                loss_val.backward()
                optimizer.step()

                train_loss += loss_val.item()
                preds = output.argmax(dim=1)
                acc += (preds == label).sum().item()
                total += label.size(0)

            train_acc = 100. * acc / total
            avg_train_loss = train_loss / len(train_data) 
        
            self.model.eval()
            val_loss, acc_val = 0.0, 0
            with torch.no_grad():
                for images, labels in val_data:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    acc_val += (preds == labels).sum().item()

            val_acc = 100. * acc_val / len(val_data.dataset)
            avg_val_loss = val_loss / len(val_data)   

            print(f"Epoch {ep+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.2f}%")
            if self.gradual:
                if ep % self.train_new_fc_step == 0:
                    self.set_prev_layer_train()
            
            if log_wandb:
                wandb.log({
                    "epoch": ep+1,
                    "train_loss": avg_train_loss,
                    "train_acc": train_acc,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc
                })

            # Early stopping and model saving
            if save_model:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), model_save_path)
                    print("Model saved at epoch {} with val loss {:.4f}".format(ep+1, avg_val_loss))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered at epoch {}".format(ep+1))                    
                        break