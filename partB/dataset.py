from torchvision import transforms,datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

def load_dataset(path,input_shape = (256,256),data_aug=True,batch_size=32):
    '''
    1. take folder path - > folder = [train,val]
    2. apply preprocessing
    3. Return train_data,val_data,test_data
    '''

    train_folder = '{}/train/'.format(path)
    test_folder = '{}/val/'.format(path)

    if not data_aug:
        data_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    elif data_aug:
        data_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])


    train_dataset_full = datasets.ImageFolder(train_folder, transform=data_transform)
    test_dataset = datasets.ImageFolder(test_folder, transform=data_transform)
    classes = train_dataset_full.classes
    
        # Collect indices per class
    class_data = {c:[] for c in classes}

    print('Loading Data ....')
    for data, label in tqdm(train_dataset_full):
        class_data[classes[label]].append((data,label))

    train_data = []
    val_data = []

    for label, data in class_data.items():
        
        split = int(0.8 * len(data))
        train_data.extend(data[:split])
        val_data.extend(data[split:])
    
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data, val_data, test_data


    
if __name__ == '__main__':
    load_dataset('./inaturalist_12K/')