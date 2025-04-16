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
                batch_normalization='yes'
                ):
        super(Classifier_Model).__init__()

        self.cn1 = nn.Conv2d(in_channels=3,out_channels=cn1_filters,kernel_size=cn1_kernel_size,padding=1)
        self.cn2 = nn.Conv2d(in_channels=3,out_channels=cn2_filters,kernel_size=cn2_kernel_size,padding=1)
        self.cn3 = nn.Conv2d(in_channels=3,out_channels=cn2_filters,kernel_size=cn2_kernel_size,padding=1)
        self.cn4 = nn.Conv2d(in_channels=3,out_channels=cn2_filters,kernel_size=cn2_kernel_size,padding=1)
        self.cn5 = nn.Conv2d(in_channels=3,out_channels=cn2_filters,kernel_size=cn2_kernel_size,padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.is_batch_normalization
        activation_map = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'gelu':nn.GELU(),
            'selu':nn.GELU(),
            'sigmoid':nn.Sigmoid(),
            'softmax':nn.Softmax(),
            'silu':nn.SiLU(),
            'mish':nn.Mish()
        }
        self.activation = activation_map[activation]
        self.is_batch_normalization = batch_normalization
        
        




