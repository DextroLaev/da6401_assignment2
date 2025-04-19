"""
Evaluates the model on the test dataset and visualizes the predictions.

This function loads the pretrained model, evaluates it on the test dataset, and logs the results to
Weights & Biases. It also visualizes the first 30 test images along with their ground truth labels, 
predictions, and a correctness flag.

Parameters:
-----------
model_path (str): The path to the model weights file (default: './models/model.pth').

Function Flow:
--------------
1. Initializes a Weights & Biases run for logging the evaluation.
2. Loads the test dataset using the `load_dataset` function.
3. Loads the model from the specified `model_path`.
4. Evaluates the model's accuracy on the test set.
5. Collects the first 30 test images, along with their ground truth labels and predictions.
6. Logs the test accuracy to Weights & Biases.
7. Visualizes the first 30 test images along with their ground truth labels, predictions, and correctness flag.
8. Creates a table in Weights & Biases with the images, labels, and predictions, and logs it as a high-resolution image grid.

Logging:
--------
- Logs the test accuracy to Weights & Biases.
- Logs a table of the first 30 test images with their corresponding ground truth and predicted labels, as well as a correctness flag, to Weights & Biases.
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
from model import PretrainedModel
from dataset import load_dataset
from config import *
import wandb
from PIL import Image

def evaluate_and_visualize(model_path='./models/best_model.pth'):    
    wandb.init(project='dl-assignment2', name='final_test_eval')

    _,_, test_loader = load_dataset('../inaturalist_12K/', batch_size=BATCH_SIZE, data_aug=DATA_AUG)


    model = PretrainedModel(train_method=TRAIN_METHOD,train_last_k_layers=TRAIN_LAST_K_LAYERS,train_new_fc_step=TRAIN_NEW_FC_STEP,out_classes=OUT_CLASSES)

    model.load_state_dict(torch.load('./models/model.pth', weights_only=True,map_location=DEVICE))

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

    for i in range(10):
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
    evaluate_and_visualize(model_path='./models/model.pth')