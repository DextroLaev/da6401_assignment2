"""
Classifier_Model: A Flexible Convolutional Neural Network (CNN) for Image Classification

This module defines the `Classifier_Model` class, which provides a configurable CNN architecture suitable
for image classification tasks. It supports dynamic filter scaling, multiple activation functions,
optional batch normalization, dropout, and a built-in training loop with early stopping and Weights & Biases (wandb) logging.

Classes
-------
Classifier_Model(nn.Module)
    Custom CNN architecture supporting various configurations and a training utility.

Example Usage
-------------
>>> model = Classifier_Model(out_classes=100, num_filters=32, activation='relu', filter_organisation='double')
>>> model.train_network(train_loader, val_loader, epochs=25, log_wandb=True, save_model=True)

Notes
-----
- The model assumes 3-channel RGB input.
- The input image dimensions should be divisible by 32 (due to 5 max-pooling layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.utils.data import DataLoader
import wandb

# Custom CNN Classifier Model with flexible architecture
class Classifier_Model(nn.Module):
    """
    A flexible convolutional neural network (CNN) classifier model for image classification tasks.
    Parameters:
        out_classes (int):              Number of output classes.
        num_filter (int):               Number of filters used with filter_organisation.
        n_dense_output_neuron (int):    Number of neurons in the dense (fully connected) layer.
        activation (str):               Activation function to use ('relu', 'tanh', 'gelu', etc.).
        filter_organisation (str):      Filter scaling strategy ('same', 'double', or 'half').
        batch_normalization (bool):     Whether to use batch normalization after convolutions.
        dropout (float):                Dropout rate applied after the dense layer.
        input_shape (tuple):            Shape of the input image (height, width).      
    """

    def __init__(self,out_classes=10,
                num_filters = 32,   
                n_dense_output_neuron = 2046,
                activation='relu',
                filter_organisation='same',
                batch_normalization=True,
                dropout=0.5,
                input_shape=(256,256)
                ):
        super(Classifier_Model,self).__init__()

        # Configure filters based on the specified filter organization        
        base = num_filters
        if filter_organisation in ['same', 'double', 'half']:
            if filter_organisation == 'same':
                filters = [base] * 5
            elif filter_organisation == 'double':
                filters = [base * (2 ** i) for i in range(5)]
            elif filter_organisation == 'half':
                filters = [max(1, base // (2 ** i)) for i in range(5)]
        else:
            raise ValueError(f"Unknown filter_organisation: {filter_organisation}")

        # Define convolutional layers with padding
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=3, padding=1)
        self.cn2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1)
        self.cn3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, padding=1)
        self.cn4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=3, padding=1)
        self.cn5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=3, padding=1)

        # Max pooling operation
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Batch normalization toggle
        self.is_batch_normalization = batch_normalization

        # Activation function mapping
        activation_map = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'gelu':nn.GELU(),
            'selu':nn.GELU(),
            'sigmoid':nn.Sigmoid(),
            'softmax':nn.Softmax(),  # Usually not used before loss in classification
            'silu':nn.SiLU(),
            'mish':nn.Mish(),
            'leaky_relu':nn.LeakyReLU()
        }

        self.activation = activation_map[activation]
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.bn5 = nn.BatchNorm2d(filters[4])

        self.flatten = nn.Flatten()        

         # Compute final height/width after conv + pooling layers
        self.final_height, self.final_width = self._calculate_output_size(input_shape)
        self.dense_input_features = filters[4] * self.final_height * self.final_width

        # Fully connected layers
        self.dense_layer = nn.Linear(self.dense_input_features, n_dense_output_neuron)
        self.output_layer = nn.Linear(n_dense_output_neuron,out_classes)
        self.dropout = nn.Dropout(dropout)

        # Move model to GPU if available
        self.to(DEVICE)
    
    # Applies convolution, optional batch norm, activation, and pooling
    def apply_conv_pass(self,x,conv_layer,batch_norm):
        """
        Applies a convolutional pass to the input tensor, including optional batch normalization,
        activation, and max pooling.
        Args:
            x (torch.Tensor):              Input tensor.
            conv_layer (nn.Conv2d):        Convolutional layer to apply.
            batch_norm (nn.BatchNorm2d):   Batch normalization layer to apply (if enabled).
        Returns:
            torch.Tensor:                  Transformed tensor after conv -> BN -> activation -> pooling.
        """

        x = conv_layer(x)
        x = self.apply(x,batch_norm)
        return x

    # Applies batch norm (if enabled), activation, and pooling
    def apply(self,x,batch_norm):
        """
        Applies batch normalization (if enabled), activation, and max pooling to the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            batch_norm (nn.BatchNorm2d): Batch normalization layer to apply.
        Returns:
            torch.Tensor: Transformed tensor.
        """

        if self.is_batch_normalization:
            x = batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x
    
    # Compute output size after all conv + pooling layers
    def _calculate_output_size(self, input_size):
        """
        Computes the final output size (height, width) after five conv + max pool operations.
        Args:
            input_size (tuple): Tuple of input image height and width.
        Returns:
            tuple: Final height and width after pooling.
        """

        height, width = input_size
        for _ in range(5):  
            height = height // 2
            width = width // 2
        return height, width

    # Forward pass through the model
    def forward(self,x):
        """
        Performs the forward pass of the CNN model.
        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).
        Returns:
            torch.Tensor: Logits for each class (B, out_classes).
        """
        x = self.apply_conv_pass(x,self.cn1,self.bn1)
        x = self.apply_conv_pass(x,self.cn2,self.bn2)
        x = self.apply_conv_pass(x,self.cn3,self.bn3)
        x = self.apply_conv_pass(x,self.cn4,self.bn4)
        x = self.apply_conv_pass(x,self.cn5,self.bn5)

        x = self.flatten(x)
        x =  self.dense_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        out = self.output_layer(x)
        return out

    # Training loop with validation and optional wandb logging
    def train_network(self, train_data, val_data,lr=1e-5, weight_decay=0.0, epochs=20,
                  model_save_path='./models/model.pth',log_wandb=False,save_model=False,early_stopping_patience=10):               
        """
        Trains the CNN model using provided training and validation datasets.
        Args:
            train_data (DataLoader): Training dataset DataLoader.
            val_data (DataLoader): Validation dataset DataLoader.
            test_data (DataLoader): Test dataset DataLoader (currently unused).
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 regularization).
            epochs (int): Number of training epochs.
            model_save_path (str): File path to save the best model.
            log (bool): Whether to log metrics to Weights & Biases (wandb).
        """
        loss_func = torch.nn.CrossEntropyLoss()        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)    
        
        best_val_loss = float('inf')
        patience_counter = 0

        for ep in range(epochs):
            self.train() 
            total_loss_train = 0
            acc = 0
            total_train = 0

            # Training loop
            for i, (img, label) in enumerate(train_data):
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                with torch.amp.autocast(device_type='cuda'):
                    output = self.forward(img)
                    loss = loss_func(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_train += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                acc += pred.eq(label.view_as(pred)).sum().item()
                total_train += label.size(0)
            
            torch.cuda.empty_cache()

            # Compute average training metrics
            train_loss_avg = total_loss_train / len(train_data)
            train_acc = 100. * acc / total_train

            # Validation phase
            self.eval()
            with torch.no_grad():
                total_loss_val = 0
                correct_val = 0

                for data, label in val_data:
                    data = data.to(DEVICE)
                    label = label.to(DEVICE)

                    out = self.forward(data)
                    total_loss_val += loss_func(out, label).item()
                    pred = out.argmax(dim=1, keepdim=True)
                    correct_val += pred.eq(label.view_as(pred)).sum().item()

                val_loss_avg = total_loss_val / len(val_data)
                val_acc = 100. * correct_val / len(val_data.dataset)
            
            torch.cuda.empty_cache()
            print(f'Epoch [{ep+1}/{epochs}], Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Log to wandb
            if log_wandb:
                wandb.log({
                    "epoch": ep+1,
                    "train_loss": train_loss_avg,
                    "train_acc": train_acc,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc
                })

            # Early stopping and model saving
            if save_model:
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    torch.save(self.state_dict(), model_save_path)
                    print("Model saved at epoch {} with val loss {:.4f}".format(ep+1, val_loss_avg))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered at epoch {}".format(ep+1))                    
                        break

