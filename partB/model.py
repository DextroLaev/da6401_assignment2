import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from partA.dataset import load_dataset
from partA.config import DEVICE
from torchvision.models import resnet50, ResNet50_Weights

class PretrainedModel(nn.Module):
    def __init__(self,out_classes=10,input_shape=(256,256),train_method='last_layer',train_last_k_layers=2,train_new_fc_step=0):
        super(PretrainedModel,self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.output_classes = out_classes
        self.loss = nn.CrossEntropyLoss()
        self.last_layer = nn.Linear(self.model.fc.in_features,self.output_classes)        
        self.train_method = train_method
        self.train_last_k_layers = train_last_k_layers 
        self.train_new_fc_step = train_new_fc_step
        self.gradual = False
        self.resnet50_layer_names = ['conv1','bn1','layer1','layer2','layer3','layer4','fc']
        self.layers_getting_trained = self.model_init()
        
        self.model.to(DEVICE)

    def model_init(self):
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
            self.resnet50_layer_names + ['new_fc']
            return train_layers+['new_fc']
        
        elif self.train_method == 'gradual':
            self.gradual = True
            self.model.fc = self.last_layer
            setattr(self.model,'new_fc',self.model.fc)
            self.resnet50_layer_names.append('new_fc')
            return ['new_fc']
    
    def set_prev_layer_train(self):
        print(self.resnet50_layer_names)
        print(self.layers_getting_trained)
        recently_set_layer_train_index = self.resnet50_layer_names.index(self.layers_getting_trained[0])
        if recently_set_layer_train_index > 0 and recently_set_layer_train_index < len(self.resnet50_layer_names):
            self.layers_getting_trained.insert(0,self.resnet50_layer_names[recently_set_layer_train_index-1])

            for name,param in self.model.named_parameters():
                if name in self.layers_getting_trained:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
    def train_network(self,train_data,val_data,test_data,epochs,lr=1e-3,batch_size=32):
        print("Training Started ..")
        train_data = DataLoader(train_data,batch_size=batch_size,shuffle=True)
        val_data = DataLoader(val_data,batch_size=batch_size,shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),lr=lr)
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
                if ep % self.train_new_fc_step:
                    self.set_prev_layer_train()
            print(self.layers_getting_trained)