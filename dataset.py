from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split


def load_dataset(path):
    '''
    1. take folder path - > folder = [train,val]
    2. apply preprocessing
    3. Return train_data,val_data,test_data
    '''

    train_folder = '{}/train/'.format(path)
    test_folder = '{}/val/'.format(path)

    transform = transforms.Compose([
        transforms.Resize((400,400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(400),  # Random crop and resize
    #     transforms.RandomHorizontalFlip(),   # Random horizontal flip
    #     transforms.RandomRotation(15),       # Random rotation
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    # ])

    train_dataset = datasets.ImageFolder(train_folder,transform=transform)
    test_dataset = datasets.ImageFolder(test_folder,transform=transform)

    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_data,val_data = random_split(train_dataset,[train_size,val_size])

    train_data = DataLoader(train_data,batch_size=32,shuffle=True)
    val_data = DataLoader(val_data,batch_size=1,shuffle=True)
    test_data = DataLoader(test_dataset,batch_size=1)

    return train_data,val_data,test_data



    
