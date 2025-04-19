# 🐦 iNaturalist 12K Image Classifier - Custom CNN with PyTorch and Pre-Trained ResNet50 Model

A configurable convolutional neural network (CNN) model for fine-grained image classification using the iNaturalist 12K dataset. This project includes:

- Custom CNN architecture with batch normalization, dropout, and filter scaling strategies
- Dataset loader with augmentation and custom train/val/test splitting
- Built-in training pipeline with early stopping and wandb logging
- Evaluation and visualization using high-resolution test predictions

---

## 🧠 Model Overview

The CNN model (`Classifier_Model`) supports:
- 5 convolutional blocks with dynamic filter scaling (`same`, `double`, `half`)
- Configurable activation functions (`ReLU`, `GELU`, `Tanh`, etc.)
- Optional batch normalization and dropout
- Fully connected layers for classification
- Training on GPU if available

---

## 📁 Directory Structure
    /iNaturallist_12K/
    /partA/
      ├── config.py # Configuration of hyperparameters and toggles 
      ├── dataset.py # Data loading and preprocessing 
      ├── model.py # Custom CNN definition 
      ├── train.py # Training script 
      ├── test.py # Evaluation + wandb table visualization 
      ├── run_swepp.py # Generates sweep run
      ├── main.py # Runs the training by taking command line arguments
      
    /partB/
      ├── config.py # Configuration of hyperparameters and toggles 
      ├── dataset.py # Data loading and preprocessing 
      ├── model.py # Contains ResNet50 training code
      ├── train.py # Training script 
      ├── test.py # Evaluation + wandb table visualization 
      ├── main.py # Runs the training by taking command line arguments

## 📂 Repository Structure

### 🔬 Experimentation Notebooks (`.ipynb` files)
- These Jupyter Notebook files were initially used to **test** and **evaluate** the neural network.  
- They are primarily for **experimentation** and performance checks.  
- These files are **not** part of the actual working code.

### ⚙️ Core Python Files (`.py` files)
These files contain the essential code for building and running the neural network:

- **`dataset.py`** 📊  
  - Handles **data loading** and **pre-processing**.  
  - Currently supports **iNaturallist_12K**.

- **`config.py`** ⚡  
  - contains the model specific configuration to run the train the respective models.


- **`test.py`** 📊  
  - This code is used evaluate the performance of the model on the testing dataset


- **`train.py`** ⚡  
  - Both the folders `pathA` and `pathB` have this file. This file is used to train the model with the default configuration
  

- **`./partA/model.py`** 🤖  
  - Implements the **CNN Model**.
  - Contains the `Classifier` class with methods like `forward`, `train_network`, etc.
  - The code is modular, allowing easy modifications to the CNN network.
  - **Note:** This file only contains the algorithm of the CNN network and is not meant to be executed directly.

- **`./partB/model.py`** 🤖  
  - Implements the **ResNet50 Model**.
  - Contains the `Pretrained` class with methods like `forward`, `train_network`, etc.
  - The code is modular, allowing easy modifications to the neural network.
  - **Note:** This file only contains the algorithm of the ResNet50 network and is not meant to be executed directly. There is three way to train the model, all the three ways are mentioned below.

- **`run_sweep_net.py`** 🤖  
  - This is used to log the selected parameters in the `wandb` platform.
  - One can check the performance of the CNN network on various hyperparameter by running this file and checking the `wandb` site.
  - Currently it logs `train loss`, `train Accuracy`, `validation loss` and `validation accuracy`.
  - One can change the `sweep_config` present inside the code, to check the performance of neural network on different hyperparameters.


- **`./partA/main.py`**
  - This script is used to train a custom convolutional neural network (CNN) model for image classification on the iNaturalist 12K dataset using configurable hyperparameters. The CNN consists of 5 convolutional blocks followed by a dense layer and an output layer. Each block contains a Conv2D -> Activation -> MaxPool2D structure. 
  - Adjustable input image size (square)
  - Configurable number and organization of filters in convolutional layers
  - Flexible activation function choices
  - Optional batch normalization and dropout
  - Dense layer with configurable number of neurons
  - Optional data augmentation
  - Support for logging and visualizing experiments with Weights & Biases (wandb)
  - Early stopping and optional model checkpoint saving
  
       **Arguments:**
      
      | Argument                     | Description |
      |------------------------------|-------------|
      | `--wandb_project` `(-wp)`     | WandB project name (required if logging). |
      | `--wandb_entity` `(-we)`      | WandB entity/user/team name (required if logging). |
      | `--input_shape` `(-input_s)`  | Size of square input image (e.g., 256). |
      | `--filter_org` `(-f_org)`     | Filter organization pattern ('same', 'double', 'half'). |
      | `--num_filters` `(-n_filters)` | Number of filters in the first convolutional layer (e.g., 64). |
      | `--data_augmentation` `(-data_aug)` | Enable/disable data augmentation (True/False). |
      | `--batch_norm` `(-bn)`        | Enable/disable batch normalization (True/False). |
      | `--dropout` `(-do)`           | Dropout rate (e.g., 0.3). |
      | `--dense_layer_neurons` `(-ls)` | Number of neurons in the dense layer. |
      | `--epochs` `(-e)`             | Number of training epochs. |
      | `--batch_size` `(-b)`         | Training batch size. |
      | `--learning_rate` `(-lr)`     | Learning rate for optimizer. |
      | `--weight_decay` `(-w_d)`     | Weight decay (L2 regularization). |
      | `--activation` `(-a)`         | Activation function ('relu', 'tanh', 'mish', 'selu', 'gelu', etc.). |
      | `--output_classes` `(--output_classes)` | Number of output classes (default: 10). |
      | `--patience_counter` `(-pc)`  | Early stopping patience. |
      | `--save_model` `(--save_model)` | Save the trained model checkpoint (True/False). |
      | `--log_wandb` `(-logw)`       | Enable logging to wandb (True/False). |


 - **`./partB/main.py`**
   
    - This script provides a command-line interface (CLI) to fine-tune a pretrained ResNet50 model on the iNaturalist 12K dataset using various training strategies. You can train the full network, freeze the earlier layers, or gradually unfreeze deeper layers. The model is trained using the `PretrainedModel` class, and the dataset is loaded via the `load_dataset` utility.
    - Supports training the full network, fine-tuning only the last layer, or gradual unfreezing of layers.
    - Configurable input image resolution, dropout, batch normalization, and dense layer size.
    - Optional data augmentation, Weights & Biases (wandb) logging, and model checkpoint saving.
    - Early stopping support with customizable patience.

      **Arguments:**
    
      | Argument                         | Description |
      |----------------------------------|-------------|
      | `--wandb_project` `(-wp)`        | WandB project name (required if logging). |
      | `--wandb_entity` `(-we)`         | WandB entity/user/team name (required if logging). |
      | `--input_shape` `(-input_s)`     | Image size (square) for input images (e.g., 224 or 256). |
      | `--data_augmentation` `(-data_aug)` | Enable/disable data augmentation (True/False). |
      | `--train_method` `(-t_method)`   | Training strategy: `last_layer`, `gradual`, or `partial`. |
      | `--train_last_k_layers` `(-t_last_k_layers)` | Number of layers (from the end) to unfreeze for `partial` training. |
      | `--train_new_fc_step` `(-t_new_fc_step)` | Step size for unfreezing layers during `gradual` training. |
      | `--epochs` `(-e)`                | Number of epochs to train the neural network. |
      | `--batch_size` `(-b)`            | Training batch size. |
      | `--learning_rate` `(-lr)`        | Learning rate for optimizer. |
      | `--weight_decay` `(-w_d)`        | Weight decay (L2 regularization). |
      | `--output_classes` `(--output_classes)` | Number of output classes (default: 10). |
      | `--patience_counter` `(-pc)`     | Early stopping patience. |
      | `--save_model` `(--save_model)`  | Whether to save the trained model after training. |
      | `--log_wandb` `(-logw)`          | Whether to log training metrics to WandB. |


## 🚀 Training the Neural Network

Before training the model you need to create a `models/` directory in both the folder i.e `/pathA` and `/pathB`
```bash
cd ./partA
mkdir models
```
```bash
cd ./partB
mkdir models
```

First install the necessary dependencies
```bash
pip3 install -r requirements.txt
```

To train the CNN (from sratch), run the following command:

```bash
cd ./partA/
python train.py
```

To train the CNN (with user input), run the following command:

```bash
python main.py -wp dl-assignment2 -we cs24s031-indian-institute-of-technology-madras --input_shape 400 -f_org double -n_filters 64 -data_aug True -bn True -do 0.3 -ls 2048 -e 20 -b 32 -lr 1e-5 -w_d 0.0005 -a gelu --output_classes 10 -pc 10 --save_model True --log_wandb True
```

After the training , run the following command to test it
```bash
python test.py
```


To train the ResNet50 (with default config), run the following command:

```bash
cd ./partB/
python train.py
```

To train the CNN (with user input), run the following command:

```bash
python main.py -wp dl-assignment2 -we cs24s031-indian-institute-of-technology-madras --input_shape 400 -data_aug True -e 20 -b 32 -lr 1e-5 -w_d 0.0005 --output_classes 10 -pc 10 --save_model True --log_wandb True -t_method last_layer
```

After the training , run the following command to test it
```bash
python test.py
```





