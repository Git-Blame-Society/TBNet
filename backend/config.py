import torch

# --- Dataset Configuration ---
TRAIN_IMAGE_PATH = './dataset/TB Dataset/Data'  # Path to training images
TEST_IMAGE_PATH = './dataset/Testing Dataset/Data'  # Path to testing images
TRAIN_CSV = './dataset/TB Dataset/Label/Label.csv'  # Training labels CSV
TEST_CSV = './dataset/Testing Dataset/Label/Label.csv'  # Testing labels CSV

# --- Processed Data Paths (for potential train/val/test splits) ---
DATA_PYTORCH_ROOT = 'data_pytorch'  # Root for organized data (optional, for future splits)
TRAIN_DIR = f'{DATA_PYTORCH_ROOT}/train'  # Training data directory
VAL_DIR = f'{DATA_PYTORCH_ROOT}/val'      # Validation data directory (if used)
TEST_DIR = f'{DATA_PYTORCH_ROOT}/test'    # Testing data directory

# --- Model & Training Parameters ---
IMG_SIZE = 224  # Image size for resizing
BATCH_SIZE = 32  # Batch size for DataLoader
EPOCHS = 10  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
NUM_CLASSES = 2  # Binary classification (0: Normal, 1: TB)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for training/inference

MEAN_IMAGENET = [0.485, 0.456, 0.406]  # ImageNet mean for normalization
STD_IMAGENET = [0.229, 0.224, 0.225]   # ImageNet std for normalization

# --- Model Saving ---
MODEL_SAVE_PATH = 'best_tb_model.pth'  # Path to save the best model

# --- Training Enhancements ---
PATIENCE_EARLY_STOPPING = 5  # Patience for early stopping (if implemented in future)