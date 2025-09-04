from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import torch
import classification
import test_convo

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

        prob = classification.PredictSymptoms(input_tensor)

        prob = float(prob)  # assume your model already returns probability
        label = 1 if prob > 0.5 else 0

        return {
            "sym_probability": prob
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-image", status_code=200)
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        prob = test_convo.test_prediction(file_location)

        os.remove(file_location)

        return {
            "image_probability": prob
        }

    except Exception as e:
        return {"error": str(e)}

