from torchvision import transforms,datasets

def load_dataset(path,input_shape = (256,256),data_aug='no'):
    '''
    1. take folder path - > folder = [train,val]
    2. apply preprocessing
    3. Return train_data,val_data,test_data
    '''

    train_folder = '{}/train/'.format(path)
    test_folder = '{}/val/'.format(path)

    if data_aug == 'no':
        data_transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    elif data_aug == 'yes':
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
    for data, label in train_dataset_full:
        class_data[classes[label]].append((data,label))

    train_data = []
    val_data = []

    for label, data in class_data.items():
        
        split = int(0.8 * len(data))
        train_data.extend(data[:split])
        val_data.extend(data[split:])

    return train_data, val_data, test_dataset


    
if __name__ == '__main__':
    load_dataset('./inaturalist_12K/')