from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

import classification
import convolution

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

        pred = 0 
    
        if(prob > 0.5):
            pred = 1 
        else:
            pred = 0

        return {"probability": prob, "prediction": pred}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily (optional)
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pred = convolution.PredictImage(file_location)

        prob = 0 

        if(pred > 0.5):
            prob = 1 
        else:
            prob = 0

        return {
            "filename": file.filename,
            "probability": prob,
            "prediction": pred  # 1 = TB, 0 = Normal
        }

    except Exception as e:
        return {"error": str(e)}
