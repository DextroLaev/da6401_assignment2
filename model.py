import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.utils.data import DataLoader
import wandb

class Classifier_Model(nn.Module):
    def __init__(self,out_classes=10,cn1_filters=32,cn1_kernel_size=3,
                cn2_filters=64,cn2_kernel_size=3,
                cn3_filters=128,cn3_kernel_size=3,
                cn4_filters=256,cn4_kernel_size=3,
                cn5_filters=512,cn5_kernel_size=3,
                filter_size = 32,   
                n_dense_output_neuron = 2046,
                activation='relu',
                filter_organisation='same',
                batch_normalization='yes',
                dropout=0.5,
                input_shape=(256,256),
                output_shape=10
                ):
        super(Classifier_Model,self).__init__()        
        base = filter_size
        if filter_organisation in ['same', 'double', 'half']:
            if filter_organisation == 'same':
                filters = [base] * 5
            elif filter_organisation == 'double':
                filters = [base * (2 ** i) for i in range(5)]
            elif filter_organisation == 'half':
                filters = [max(1, base // (2 ** i)) for i in range(5)]
        else:
            raise ValueError(f"Unknown filter_organisation: {filter_organisation}")

        self.cn1 = nn.Conv2d(in_channels=3, out_channels=filters[0], kernel_size=cn1_kernel_size, padding=1)
        self.cn2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=cn2_kernel_size, padding=1)
        self.cn3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=cn3_kernel_size, padding=1)
        self.cn4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=cn4_kernel_size, padding=1)
        self.cn5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=cn5_kernel_size, padding=1)

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
            'leaky_relu':nn.LeakyReLU()
        }
        self.activation = activation_map[activation]
        if batch_normalization:
            self.bn1 = nn.BatchNorm2d(filters[0])
            self.bn2 = nn.BatchNorm2d(filters[1])
            self.bn3 = nn.BatchNorm2d(filters[2])
            self.bn4 = nn.BatchNorm2d(filters[3])
            self.bn5 = nn.BatchNorm2d(filters[4])

        self.flatten = nn.Flatten()        
        self.final_height, self.final_width = self._calculate_output_size(input_shape)
        self.dense_input_features = filters[4] * self.final_height * self.final_width

        self.dense_layer = nn.Linear(self.dense_input_features, n_dense_output_neuron)
        self.output_layer = nn.Linear(n_dense_output_neuron,out_classes)
        self.dropout = nn.Dropout(dropout)
        self.to(DEVICE)
    
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

    def train_network(self, train_data, val_data, test_data, batch_size=32, lr=1e-5, weight_decay=0.0, epochs=1000,
                  model_save_path='./models/model.pth', early_stopping_patience=10,log=False):        
        
        train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)
        val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True)

        loss_func = torch.nn.CrossEntropyLoss()        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)    

        best_val_loss = float('inf')
        patience_counter = 0

        for ep in range(epochs):
            self.train() 
            total_loss_train = 0
            acc = 0
            total_train = 0

            for i, (img, label) in enumerate(train_data):
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                output = self.forward(img)
                loss = loss_func(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_train += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                acc += pred.eq(label.view_as(pred)).sum().item()
                total_train += label.size(0)

            del img, label, output, loss, pred
            torch.cuda.empty_cache()
            train_loss_avg = total_loss_train / len(train_data)
            train_acc = 100. * acc / total_train
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

            del data,label,out,pred,total_loss_val
            torch.cuda.empty_cache()
            print(f'Epoch [{ep+1}/{epochs}], Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Log to wandb
            if log:
                wandb.log({
                    "epoch": ep+1,
                    "train_loss": train_loss_avg,
                    "train_acc": train_acc,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc
                })

            # Early stopping and model saving
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

        wandb.finish()

