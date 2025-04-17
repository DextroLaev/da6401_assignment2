import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from dataset import load_dataset
from config import DEVICE
from torchvision.models import resnet50, ResNet50_Weights


train_data, val_data, test_data = load_dataset('./inaturalist_12K/')

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)


model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)


model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)

    # 🔵 Validation
    model.eval()
    val_loss, correct_val = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()

    val_acc = 100. * correct_val / len(val_loader.dataset)
    avg_val_loss = val_loss / len(val_loader)   

    print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.2f}%")

    scheduler.step()

wandb.finish()
