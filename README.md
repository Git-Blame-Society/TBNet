# TBNet : Tuberculosis Detection using CNNs 🩺🫁

TBNet is a deep learning project that detects **tuberculosis (TB)** from **chest X‑ray images** and also includes a **symptom‑based classifier** for quick runtime predictions. It combines a **PyTorch backend (FastAPI)** with a **Next.js frontend** to provide an interactive interface for researchers and learners.

---

## ⚠️ Disclaimer

This project is developed **only for educational and research purposes**. It is **not intended for clinical or diagnostic use**. Real‑world AI tools for medical diagnosis undergo extensive training, validation, and regulatory approval before deployment in healthcare. Please **do not use this project for actual medical decisions**.

---

## 🚀 Features

* **Two Models**

  * **Symptom Classifier** – lightweight MLP, trains at runtime
  * **Convolutional Image Classifier** – CNN for chest X‑rays
* **End‑to‑End Pipeline**

  * Train CNN on chest X‑ray dataset
  * Automatically save best model (`best_tb_model.pth`)
  * Upload symptoms or X‑ray images via frontend → FastAPI backend → prediction
* **Frontend**: Next.js, Tailwind CSS, shadcn/ui
* **Backend**: FastAPI (Python) serving PyTorch models

---

## 📂 Dataset

* **Original dataset**: [Tuberculosis Chest X‑ray Dataset (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* **Preprocessed + labeled dataset (ready‑to‑use splits)**: [Google Drive Link](https://drive.google.com/drive/folders/1MpTOFgSre2V4ueVPq1yQX2gJ0zedM1_e?usp=sharing)

> **Credit:** If you use this project, please credit the Kaggle dataset and original authors.

---

## 🧠 Model Architectures & Rationale

### 1) Convolutional Image Classifier (CNN)

**File:** `convolution.py`
**Task:** Binary classification (TB vs. Normal) from chest X‑ray.

**Architecture (PyTorch):**

```python
class TBClassifier(nn.Module):
    def __init__(self):
        super().__init__()
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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Why these choices?**

* **224×224 input** with **ImageNet mean/std normalization** stabilizes training and is a common convention; even without transfer learning, these stats help gradients scale reasonably.
* **3×3 convs + ReLU + MaxPool** are a proven baseline for learning local features (edges, textures) that scale up to higher‑level patterns.
* **Dropout(0.25)** before the final layer reduces overfitting on relatively small medical datasets.
* **`CrossEntropyLoss`** is the standard for multi‑class classification with logits; it combines `LogSoftmax` + NLL in a stable way.
* **`Adam(lr=1e‑3)`** offers fast convergence without much tuning.

**Transforms (input preprocessing):**

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

* Resize ensures consistent tensor shapes; normalization improves optimization.

**Prediction API (helper):**

```python
def PredictImage(image_path):
    model = TBClassifier().to(device)
    model.load_state_dict(torch.load("best_tb_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        probs = F.softmax(output, dim=1)
        prob_tb = probs[0][1].item()  # class 1 = TB
        return prob_tb
```

* **Softmax** converts logits to probabilities for the two classes; we report **P(TB)**.

**Default hyperparameters:** `epochs=10`, `criterion=CrossEntropyLoss()`, `optimizer=Adam(lr=0.001)`.

> **Tip (optional improvement):** A validation split can be added and the best model can be tracked by **validation accuracy** or **AUROC**; can also try **data augmentation** (random flips/rotations) or a pretrained backbone (e.g., ResNet) for stronger baselines.

---

### 2) Symptom Classifier (MLP)

**File:** `classification.py`
**Task:** Binary probability of TB from a vector of symptoms.

**Architecture (PyTorch):**

```python
class Classification(nn.Module):
    def __init__(self, input_size, hidden1=16, hidden2=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # probability
        return x
```

**Why these choices?**

* **Small MLP** keeps runtime training fast for tabular symptom inputs.
* **Sigmoid activations** throughout maintain outputs in \[0,1] and are simple for small networks.
* **`BCELoss`** matches a probability output (`sigmoid` on the final neuron) for binary classification.
* **`Adam(lr=1e‑2)`, `epochs=100`** converge quickly on lightweight inputs.

> **Tip (optional improvement):** For numerical stability, many projects use `BCEWithLogitsLoss` **without** a final sigmoid (the loss applies the sigmoid internally). Either approach works; Current setup is for simplicity.

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, PyTorch
* **Frontend**: Next.js v15.5.2, Tailwind CSS, shadcn/ui
* **Languages/Runtime**: Python 3.12.10, Node v22.16.0

---

## ⚙️ Setup & Installation

### 1) Clone the repository

```bash
git clone https://github.com/Git-Blame-Society/TBNet.git
cd TBNet
```

### 2) Backend setup (FastAPI + PyTorch)

Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Install dependencies (from `requirements.txt`):

```bash
pip install -r requirements.txt
```

**Key packages**: `torch`, `torchvision`, `fastapi`, `uvicorn`, `pillow`, `numpy`.

**Download dataset** and place it at the **same level** as `convolution.py`:

```
backend/
├── convolution.py
├── test_convo.py
├── classification.py
├── main.py
└── dataset/            # ← put dataset here
```

**Train the CNN**:

```bash
python convolution.py
```

* Saves best weights to `best_tb_model.pth` (same directory).

**Run FastAPI backend**:

```bash
uvicorn main:app --reload
```

* Backend: `http://127.0.0.1:8000`

### 3) Frontend setup (Next.js)

```bash
cd frontend
npm install
npm run dev
```

* Frontend: `http://localhost:3000`
* Connects to backend at `http://127.0.0.1:8000`

> **Optional**: Configure the API base via env var (`frontend/.env.local`):
>
> ```env
> NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
> ```

---

## 🔮 Usage Flow

1. Train CNN via `convolution.py` → best model saved as `best_tb_model.pth`.
2. Start backend: `uvicorn main:app --reload`.
3. Start frontend: `npm run dev`.
4. Use the UI to upload **symptoms** (JSON array) or an **X‑ray image** → get probabilities.

---

## 🔌 API Endpoints (FastAPI)

* **GET /** → Health check

  * **Response**: `{ "value": "Local FastAPI running" }`

* **POST /upload-symptoms** → Predict TB probability from symptom features

  * **Body (JSON)**: `{ "result": [f1, f2, ..., fn] }`
  * **Response**: `{ "sym_probability": 0.42 }`

* **POST /upload-image** → Predict TB probability from X‑ray image

  * **Body (multipart/form-data)**: `file=@your_image.jpg`
  * **Response**: `{ "image_probability": 0.87 }`

**cURL examples:**

```bash
curl -X POST http://127.0.0.1:8000/upload-symptoms \
  -H "Content-Type: application/json" \
  -d '{"result":[0,1,0,1,0,0,1]}'

curl -X POST http://127.0.0.1:8000/upload-image \
  -F "file=@sample_xray.jpg"
```




https://github.com/user-attachments/assets/4a56cda5-4f0b-45da-bbd1-e49561212596




---

## 📁 Project Structure

```bash
TBNet/
├── backend/
│   ├── dataset/            # Chest X‑ray dataset (user downloads separately)
│   ├── best_tb_model.pth   # Saved best CNN model (after training)
│   ├── classification.py   # Symptom classifier (MLP)
│   ├── convolution.py      # Train CNN on chest X‑rays
│   ├── main.py             # FastAPI backend
│   ├── requirements.txt
│   └── test_convo.py       # Test CNN with pretrained model
├── frontend/               # Next.js frontend
│   ├── node_modules/       # (created after npm install)
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ...other config files
├── .gitignore
└── README.md
```

---

## 🧪 Evaluation (optional)

* Suggested metrics: **Accuracy**, **Precision/Recall**, **F1**, **ROC‑AUC** on a held‑out validation set.
* For imbalanced data, also report **confusion matrix** and **PR‑AUC**.

---

## 🧷 Reproducibility (optional)

Set seeds in `convolution.py` if you want deterministic runs:

```python
import torch, random, numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 🙌 Credits

* Dataset: [Tuberculosis Chest X‑ray Dataset (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* Libraries: PyTorch, FastAPI, Next.js, Tailwind CSS, shadcn/ui

---

## 📌 Future Improvements

* Deploy model (AWS/GCP/Render/HuggingFace Spaces)
* Data augmentation, transfer learning (ResNet/EfficientNet)
* Better monitoring (tensorboard, confusion matrix, ROC curve)
* UI/UX improvements for usability

---

## 📜 License

MIT License – feel free to fork, modify, and use with attribution.
