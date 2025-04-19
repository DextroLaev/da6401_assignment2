import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_CLASSES = 10

DATA_AUG = False
BATCH_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 25
LOG_WANDB = True
SAVE_MODEL = True
EARLY_STOPPING_PATIENCE = 10

TRAIN_METHOD = 'last_layerpu'
TRAIN_LAST_K_LAYERS = 3
TRAIN_NEW_FC_STEP = 2
