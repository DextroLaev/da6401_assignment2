import torch
from dataset import load_dataset
from model import Classifier_Model
import matplotlib.pyplot as plt
from config import *

def train_network(model,train_data,test_data,val_data,epochs=1000,lr=1e-5):

    loss_func = torch.nn.CrossEntropyLoss()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)    

    for ep in range(epochs):
        model.train() 
        for i, (img, label) in enumerate(train_data):
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            output = model(img)
            
            loss = loss_func(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss_v = 0
            acc = 0
            for data, label in val_data:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                out = model(data)
                total_loss_v += loss_func(out, label).item()

                pred = out.argmax(dim=1, keepdim=True)
                acc += pred.eq(label.view_as(pred)).sum().item()

            accuracy = 100. * acc / len(val_data.dataset)

            print(f'Epoch [{ep+1}/{epochs}], Validation Loss: {total_loss_v:.4f}, Validation Accuracy: {accuracy:.2f}%')

        model.train()


if __name__ == '__main__':
    train_data,val_data,test_data = load_dataset('./inaturalist_12K')

    model = Classifier_Model(out_classes=10,cn1_filters=32,cn1_kernel_size=3,
                             cn2_filters=64,cn2_kernel_size=3,cn3_filters=128,cn3_kernel_size=3,
                             cn4_filters=256,cn4_kernel_size=3,cn5_filters=512,cn5_kernel_size=3,
                             n_dense_output_neuron=2048,activation='gelu',filter_organisation='same',
                             batch_normalization='yes',dropout=0.5)
    
    train_network(model,train_data,test_data,val_data,1000,0.1e-5)


