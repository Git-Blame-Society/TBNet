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

# Load your first CSV file (dataset_1) - Optional if you need it for any other purpose
url_1 = "./dataset/TB Dataset/Label/symptoms.csv"  #path to symptoms data CSV file
dataset = pd.read_csv(url_1)

# Clean the columns in both CSV files
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(" ", "_")

X_symptoms = []
y_symptoms = []

for index, row in dataset.iterrows():
    X_symptoms.append(row[['fever_for_two_weeks', 'coughing_blood',
                  'sputum_mixed_with_blood', 'night_sweats', 'chest_pain',
                  'back_pain_in_certain_parts', 'shortness_of_breath', 'weight_loss', 'body_feels_tired', 'lumps_that_appear_around_the_armpits_and_neck',
                  'cough_and_phlegm_continuously_for_two_weeks_to_four_weeks', 'swollen_lymph_nodes', 'loss_of_appetite']].tolist())

for index, row in dataset.iterrows():
  y_symptoms.append(row['diagnosis'])

# Convert to tensors
X_symptoms_tensor = torch.tensor(X_symptoms, dtype=torch.float32)   # shape: [num_samples, num_features]
y_symptoms_tensor = torch.tensor(y_symptoms, dtype=torch.float32).unsqueeze(1)  # shape: [num_samples, 1]


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

model = Classification(input_size=X_symptoms_tensor.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
  optimizer.zero_grad()
  output = model(X_symptoms_tensor)
  loss = criterion(output, y_symptoms_tensor)
  loss.backward()
  optimizer.step()

  if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def PredictSymptoms(input):
    with torch.no_grad():
        prob = model(input).item()
        return prob
