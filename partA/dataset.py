"""
    Dataset Loader for iNaturalist 12K (or similar folder-structured datasets)

    This module defines the `load_dataset` function, which loads images from a specified dataset path
    structured using torchvision's `ImageFolder` format. It supports optional data augmentation and returns
    PyTorch DataLoaders for training, validation, and testing.

    Functions
    ---------
    load_dataset(path, input_shape=(256, 256), data_aug=True, batch_size=32)
        Loads image data from a folder-structured dataset and returns DataLoaders for train, val, and test sets.

    Details
    -------
    Expected folder structure:
        path/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── val/
            ├── class1/
            ├── class2/
            └── ...

    Parameters
    ----------
    path : str
        Root path of the dataset. Must contain 'train' and 'val' subdirectories with class folders inside.
    input_shape : tuple of int, optional
        Size to resize each image to, as (height, width). Default is (256, 256).
    data_aug : bool, optional
        Whether to apply data augmentation (random horizontal flip and rotation) to training images. Default is True.
    batch_size : int, optional
        Number of samples per batch in the returned DataLoaders. Default is 32.

    Returns
    -------
    train_data : torch.utils.data.DataLoader
        DataLoader for the training portion of the training dataset (80% split per class).
    val_data : torch.utils.data.DataLoader
        DataLoader for the validation portion of the training dataset (20% split per class).
    test_data : torch.utils.data.DataLoader
        DataLoader for the testing dataset (loaded from 'val' folder).

    Note
    ----
    This function manually splits the training data class-wise into training and validation subsets.
    The `val/` directory is treated as the test set.
"""


from torchvision import transforms,datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

def load_dataset(path,input_shape = (256,256),data_aug=True,batch_size=32):
    """
    Loads and preprocesses an image dataset from a given path using torchvision's ImageFolder.

    The dataset directory should contain two subfolders: 'train' and 'val', each with subdirectories
    for every class. The function applies preprocessing and optional data augmentation, then splits
    the training data into training and validation sets.
    Parameters
    ----------
    path : str
        Path to the dataset root directory containing 'train' and 'val' folders.
    input_shape : tuple of int, optional
        The desired image size (height, width) after resizing. Default is (256, 256).
    data_aug : bool, optional
        If True, applies data augmentation (random horizontal flip and rotation) to training images.
        Default is True.
    batch_size : int, optional
        Number of samples per batch to load. Default is 32.
    Returns
    -------
    train_data : torch.utils.data.DataLoader
        DataLoader for the training split of the training dataset.
    val_data : torch.utils.data.DataLoader
        DataLoader for the validation split of the training dataset.
    test_data : torch.utils.data.DataLoader
        DataLoader for the validation dataset (from 'val' folder, used as test set).
    """

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
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_data, val_data, test_data


    
if __name__ == '__main__':
    load_dataset('./inaturalist_12K/')
