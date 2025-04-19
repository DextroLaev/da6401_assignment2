import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Classifier_Model
from dataset import load_dataset
from config import *
import wandb
from PIL import Image

def evaluate_and_visualize(model_path='./models/best_model.pth'):
    """
    Loads a trained CNN model and evaluates its performance on the test dataset. 
    Computes the test accuracy and visualizes predictions using Weights & Biases (wandb).

    Args:
        model_path (str): 
            Path to the saved model weights file (default: './models/best_model.pth').

    Behavior:
        - Initializes a Weights & Biases (wandb) run for logging.
        - Loads the test dataset using predefined configurations.
        - Loads the CNN model architecture and its saved weights.
        - Performs inference on the test set (up to 30 samples).
        - Calculates test accuracy and logs it to wandb.
        - Displays a table of 10 high-resolution test sample predictions:
            - Image (visualized),
            - Ground Truth label,
            - Model Prediction,
            - Whether the prediction was correct.

    Notes:
        - Assumes test dataset and class names are accessible via `load_dataset`.
        - Converts images to high-resolution PIL format for better wandb rendering.
        - Normalization is reversed for visualization purposes.
        - Requires the `model.py`, `config.py`, and `dataset.py` files to define the model architecture, configuration, and dataset loading respectively.
        - Logs and finalizes the wandb session upon completion.

    Example usage:
        >>> evaluate_and_visualize(model_path='./models/model.pth')
    """ 

    
    wandb.init(project='dl-assignment2', name='final_test_eval')

    _,_, test_loader = load_dataset('../inaturalist_12K/', batch_size=BATCH_SIZE, data_aug=DATA_AUG)


    model = Classifier_Model(out_classes=OUT_CLASSES, n_dense_output_neuron=DENSE_NEURONS, activation=ACTIVATION,filter_organisation=FILTER_ORGANISATION,
                            num_filters=NUM_FILTERS, batch_normalization=BATCH_NORMALIZATION,
                            dropout=DROPOUT)

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
