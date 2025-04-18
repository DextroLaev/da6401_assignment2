import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Classifier_Model
from dataset import load_dataset
from config import *
import wandb
from PIL import Image

def evaluate_and_visualize(model_path='./models/best_model.pth'):    
    wandb.init(project='dl-assignment2', name='final_test_eval')

    train_data, val_data, test_loader = load_dataset('../inaturalist_12K/', batch_size=32, data_aug=False)


    model = Classifier_Model(out_classes=10, n_dense_output_neuron=4192, activation='silu',filter_organisation='double',num_filters=64, batch_normalization=True,dropout=0.5)

    # model.train_network(train_data,val_data,lr=2e-5, weight_decay=0.0, epochs=20,
    #               model_save_path='./models/model.pth',log=False)
    
    model.eval()

    correct = 0
    total = 0
    all_images = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_images.append(images.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if len(all_preds) >= 30:
                break  

    test_acc = 100 * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')
    wandb.log({"Test Accuracy": test_acc})

    images = torch.cat(all_images)[:30]
    preds = all_preds[:30]
    labels = all_labels[:30]
    class_names = test_loader.dataset.classes

    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Correct"])

    for i in range(30):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])).clip(0, 1)

        # Convert to high-res PIL Image for better wandb rendering
        img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((310, 310))

        label = labels[i]
        pred = preds[i]
        correct_flag = label == pred
        wandb_img = wandb.Image(img_pil, caption=f"GT: {class_names[label]}, Pred: {class_names[pred]}")

        table.add_data(wandb_img, class_names[label], class_names[pred], correct_flag)

    wandb.log({"Test Predictions Table (High-Res 10x3 Grid)": table})
    wandb.finish()

if __name__ == '__main__':
    evaluate_and_visualize(model_path='./models/best_model.pth')