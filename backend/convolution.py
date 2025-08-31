import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from torch import optim

# Set the path to your dataset
image_dataset_path = "./dataset/TB Dataset/Data/"  # Update with your dataset path

# Load your second CSV file (dataset_2 - image labels)
url_2 = "./dataset/TB Dataset/Label/Label.csv"      #path to image label CSV file
dataset_2 = pd.read_csv(url_2)

# Clean the columns in both CSV files
dataset_2.columns = dataset_2.columns.str.strip().str.lower().str.replace(" ", "_")
dataset_2["name"] = dataset_2["name"].astype(str) + ".png"

# Rename columns in dataset_2 to match the expected format
dataset_2.rename(columns={"name": "filename", "label": "label"}, inplace=True)

# Check if path exists
print("Dataset path:", image_dataset_path)
print("Is the dataset path valid?", os.path.exists(image_dataset_path))

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to a consistent size (224x224)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (scaled [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet mean/std)
])

# Dataset Class to load images and labels
class TBImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, csv_data=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Map the filenames from the folder and labels from the CSV
        self.image_paths = [os.path.join(image_dir, filename) for filename in csv_data['filename']]
        self.labels = [label for label in csv_data['label']]  # Use labels directly from the CSV

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image using PIL
        image = Image.open(image_path).convert("RGB")

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label


# Initialize the dataset with the second CSV data for labels
image_dataset = TBImageDataset(
    image_dir=image_dataset_path,
    transform=transform,
    csv_data=dataset_2  # Pass the second CSV dataset for image labels
)

# Check how many samples are loaded
print(f"Number of samples in dataset: {len(image_dataset)}")

# Create DataLoaders
image_data = DataLoader(image_dataset, batch_size=7, shuffle=True)

# Test one batch of images
for images, labels in image_data:
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels: {labels}")
    break  # Just show the first batch

all_images = []
all_labels = []

for images, labels in image_data:
    all_images.extend(images)
    all_labels.extend(labels)

X_tensor_image = torch.stack(all_images, dim=0)      # images are already batched
Y_tensor_image = torch.stack(all_labels, dim=0)    # stack scalars into 1D tensor

# --- Model (same as yours) ---
class TBClassifier(nn.Module):
    def __init__(self):
        super(TBClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # After 2 poolings: 224 -> 112 -> 56
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = TBClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Prepare dataset tensors ---
X_tensor_image = torch.stack(all_images, dim=0)        # shape [N, 3, 224, 224]
Y_tensor_image = torch.tensor(all_labels, dtype=torch.long)  # shape [N]
print("Dataset:", X_tensor_image.shape, Y_tensor_image.shape)

# --- Training with mini-batches ---
batch_size = 32
epochs = 10

num_samples = X_tensor_image.size(0)

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, num_samples, batch_size):
        X_batch = X_tensor_image[i:i+batch_size].to(device)
        Y_batch = Y_tensor_image[i:i+batch_size].to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

    epoch_loss = running_loss / (num_samples / batch_size)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("✅ Training finished")

def PredictImage(image_path):
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)
        probs = F.softmax(output, dim=1)  # [batch, 2]
        prob_tb = probs[0][1].item()      # probability for TB (class 1)

        label = 1 if prob_tb > 0.5 else 0
        return prob_tb, label

