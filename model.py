import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_Model(nn.Module):
    def __init__(self,out_classes=10,cn1_filters=32,cn1_kernel_size=3,
                cn2_filters=64,cn2_kernel_size=3,
                cn3_filters=128,cn3_kernel_size=3,
                cn4_filters=256,cn4_kernel_size=3,
                cn5_filters=512,cn5_kernel_size=3,
                n_dense_input_neuron = 1000,
                n_dense_output_neuron = 2046,
                activation='relu',
                filter_organisation='same',
                batch_normalization='yes',
                dropout=0.5,
                input_shape=(400,400)
                ):
        super(Classifier_Model,self).__init__()

        self.cn1 = nn.Conv2d(in_channels=3,out_channels=cn1_filters,kernel_size=cn1_kernel_size,padding=1)
        self.cn2 = nn.Conv2d(in_channels=cn1_filters,out_channels=cn2_filters,kernel_size=cn2_kernel_size,padding=1)
        self.cn3 = nn.Conv2d(in_channels=cn2_filters,out_channels=cn3_filters,kernel_size=cn3_kernel_size,padding=1)
        self.cn4 = nn.Conv2d(in_channels=cn3_filters,out_channels=cn4_filters,kernel_size=cn4_kernel_size,padding=1)
        self.cn5 = nn.Conv2d(in_channels=cn4_filters,out_channels=cn5_filters,kernel_size=cn5_kernel_size,padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.is_batch_normalization = batch_normalization
        activation_map = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'gelu':nn.GELU(),
            'selu':nn.GELU(),
            'sigmoid':nn.Sigmoid(),
            'softmax':nn.Softmax(),
            'silu':nn.SiLU(),
            'mish':nn.Mish(),
            'leaky_rule':nn.LeakyReLU()
        }
        self.activation = activation_map[activation]
        if batch_normalization:
            self.bn1 = nn.BatchNorm2d(cn1_filters)
            self.bn2 = nn.BatchNorm2d(cn2_filters)
            self.bn3 = nn.BatchNorm2d(cn3_filters)
            self.bn4 = nn.BatchNorm2d(cn4_filters)
            self.bn5 = nn.BatchNorm2d(cn5_filters)

        self.flatten = nn.Flatten()        
        self.final_height, self.final_width = self._calculate_output_size(input_shape)
        self.dense_input_features = cn5_filters * self.final_height * self.final_width

        self.dense_layer = nn.Linear(self.dense_input_features, n_dense_output_neuron)
        self.output_layer = nn.Linear(n_dense_output_neuron,out_classes)
        self.dropout = nn.Dropout(dropout)
    
    def apply_conv_pass(self,x,conv_layer,batch_norm):
        x = conv_layer(x)
        x = self.apply(x,batch_norm)
        return x

    def apply(self,x,batch_norm):
        if self.is_batch_normalization:
            x = batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x
    
    def _calculate_output_size(self, input_size):
        height, width = input_size
        for _ in range(5):  
            height = height // 2
            width = width // 2
        return height, width

    def forward(self,x):
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
