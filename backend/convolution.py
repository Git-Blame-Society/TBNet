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

# --- Dataset Class to load images and labels ---
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

# --- Model ---
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

# --- Define the image transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to a consistent size (224x224)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (scaled [0, 1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet mean/std)
])

# --- Cleaning the coloums of labels ---
def clean_labels(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["name"] = df["name"].astype(str) + ".png"

    # Renaming columns in dataset_2 to match the expected format
    df.rename(columns={"name": "filename", "label": "label"}, inplace=True)
    return df

# --- Prediction Function ---
def PredictImage(image_path):
    model = TBClassifier().to(device)
    model.load_state_dict(torch.load("best_tb_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)
        probs = F.softmax(output, dim=1)  # [batch, 2]
        prob_tb = probs[0][1].item()      # probability for TB (class 1)
        
        return prob_tb

# --- Main Code ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    # Setting up the path to the datasets
    train_image_path = "./dataset/TB Dataset/Data"
    test_image_path = "./dataset/Testing Dataset/Data"

    train_csv = "./dataset/TB Dataset/Label/Label.csv"
    test_csv = "./dataset/Testing Dataset/Label/Label.csv"

    # Loading the CSV Label files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = clean_labels(train_df)
    test_df = clean_labels(test_df)

    # Check if path exists
    print("Train Dataset path:", train_image_path)
    print("Is the dataset path valid?", os.path.exists(train_image_path))
    print("Test Dataset path:", test_image_path)
    print("Is the dataset path valid?", os.path.exists(test_image_path))

    # Initialize the dataset with the CSV data for labels
    train_dataset = TBImageDataset(
        image_dir=train_image_path,
        transform=transform,
        csv_data=train_df  # Pass the second CSV dataset for image labels
    )
    test_dataset = TBImageDataset(
        image_dir=test_image_path,
        transform=transform,
        csv_data=test_df  # Pass the second CSV dataset for image labels
    )

    # Check how many samples are loaded
    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory = True)

    print(f"Train Samples: {len(train_dataset)}, Test Samples: {len(test_dataset)}")

    # Test one batch of images
    for images, labels in test_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels: {labels}")
        break  # Just show the first batch

    # --- Training Setup ---

    model = TBClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_acc = 0.0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # looping directly over batches from Dataloader
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- Testing or Validation ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        print(f"Epoch [{epoch+1} / {epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% \n Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_tb_model.pth")
            print(f"✅ Saved new best model with acc {best_acc:.2f}%")

    print("✅ Training finished")
