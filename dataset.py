from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split


def load_dataset(path,data_aug='no'):
    '''
    1. take folder path - > folder = [train,val]
    2. apply preprocessing
    3. Return train_data,val_data,test_data
    '''

    train_folder = '{}/train/'.format(path)
    test_folder = '{}/val/'.format(path)

    base_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    if data_aug == 'yes':
        train_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop((400, 400), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = base_transform


    train_dataset_full = datasets.ImageFolder(train_folder, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_folder, transform=train_transform)

    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


    
