from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_id = "19zZLmLJ9KVqPHdmXhHLJG1qZNBreswTD"

# Use the proper export format to get CSV
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"

# Load into pandas
dataset = pd.read_csv(url)

# Clean column names
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(" ", "_")

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

@app.get("/")
def main():
    return {"value": "Local FastAPI running"}

@app.post("/upload-symptoms", status_code=200)
def uploadSymptoms(payload: dict):
    try:
        # Extract the array from the dict
        features = payload.get("result")
        
        # Make sure it's a list of numbers
        input_tensor = torch.tensor([features], dtype=torch.float32)  # wrap in [ ] so shape = (1, num_features)

        with torch.no_grad():
            prob = model(input_tensor).item()
            prediction = 1 if prob >= 0.5 else 0

        print("Predicted Probability TB:", prob)
        print("Predicted Class:", prediction)

        return {"probability": prob, "prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
