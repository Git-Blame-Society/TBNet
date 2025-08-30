import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F

# Mount Google Drive
# drive.mount('/content/drive/')

# Set the path to your dataset
image_dataset_path = "./dataset/TB Dataset/Data/"  # Update with your dataset path

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


# Load your first CSV file (dataset_1) - Optional if you need it for any other purpose
url_1 = "./dataset/TB Dataset/Label/symptoms.csv"  #path to symptoms data CSV file
dataset = pd.read_csv(url_1)

# Load your second CSV file (dataset_2 - image labels)
url_2 = "./dataset/TB Dataset/Label/Label.csv"      #path to image label CSV file
dataset_2 = pd.read_csv(url_2)

# Clean the columns in both CSV files
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(" ", "_")
dataset_2.columns = dataset_2.columns.str.strip().str.lower().str.replace(" ", "_")
dataset_2["name"] = dataset_2["name"].astype(str) + ".png"

# Rename columns in dataset_2 to match the expected format
dataset_2.rename(columns={"name": "filename", "label": "label"}, inplace=True)

# Print dataset_1 and dataset_2 if you want to see the contents
print("Dataset 1:\n", dataset.head())
print("Dataset 2:\n", dataset_2.head())

# Initialize the dataset with the second CSV data for labels
image_dataset = TBImageDataset(
    image_dir=image_dataset_path,
    transform=transform,
    csv_data=dataset_2  # Pass the second CSV dataset for image labels
)

# Check how many samples are loaded
print(f"Number of samples in dataset: {len(image_dataset)}")

# Create train-test split (80% train, 20% test)
# train_size = int(0.8 * len(image_dataset))
# test_size = len(image_dataset) - train_size
# train_dataset, test_dataset = random_split(image_dataset, [train_size, test_size])

# Create DataLoaders
image_data = DataLoader(image_dataset, batch_size=7, shuffle=True)

# Test one batch of images
for images, labels in image_data:
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels: {labels}")
    break  # Just show the first batch

X = []
y = []

for index, row in dataset.iterrows():
    X.append(row[['fever_for_two_weeks', 'coughing_blood',
                  'sputum_mixed_with_blood', 'night_sweats', 'chest_pain',
                  'back_pain_in_certain_parts', 'shortness_of_breath', 'weight_loss', 'body_feels_tired', 'lumps_that_appear_around_the_armpits_and_neck',
                  'cough_and_phlegm_continuously_for_two_weeks_to_four_weeks', 'swollen_lymph_nodes', 'loss_of_appetite']].tolist())

# use checkboxs UI to check if the patient has the above symptoms

for index, row in dataset.iterrows():
  y.append(row['diagnosis'])

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)   # shape: [num_samples, num_features]
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: [num_samples, 1]


class Classification(nn.Module):
    def __init__(self, input_size, hidden1=16, hidden2=8):
        super().__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden1)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        # Output layer
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        # Pass through hidden layers with ReLU
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # Final layer with sigmoid for binary classification
        x = torch.sigmoid(self.fc3(x))
        return x

model = Classification(input_size=X_tensor.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
  optimizer.zero_grad()
  output = model(X_tensor)
  loss = criterion(output, y_tensor)
  loss.backward()
  optimizer.step()

  if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Example test patient symptoms:
# fever=1, coughing_blood=0, sputum_blood=1, night_sweats=0, chest_pain=1, back_pain=0, shortness_of_breath=1
test_input = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

with torch.no_grad():  # disable gradients during inference
    prob = model(test_input).item()
    print("Predicted Probability TB:", prob)
    print("Predicted Class:", 1 if prob >= 0.5 else 0)

result = None

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
