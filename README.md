# TBNet : Tuberculosis Detection using CNNs 🩺🫁

TBNet is a deep learning project that detects **tuberculosis (TB) from chest X-ray images** and also includes a **symptom-based classifier** for quick runtime predictions.
It combines a **PyTorch backend (FastAPI)** with a **Next.js frontend** to provide an interactive interface for doctors/researchers.

---

## ⚠️ Disclaimer
This project is developed **only for educational and research purposes**.
It is **not intended for clinical or diagnostic use**.
Real-world AI tools for medical diagnosis undergo extensive training, validation, and regulatory approval before deployment in healthcare.

Please **do not use this project for actual medical decisions**.

---

## 🚀 Features
- **Two Models**:
  - Symptom Classifier (lightweight MLP, trains at runtime)
  - Convolutional Image Classifier (CNN for chest X-rays)
- **End-to-End Pipeline**:
  - Train CNN on chest X-ray dataset
  - Save best model automatically (`best_tb_model.pth`)
  - Upload symptoms or X-ray images through frontend → FastAPI backend → Model prediction
- **Frontend**: Built with Next.js, Tailwind, and Shadcn UI
- **Backend**: FastAPI for model serving (symptoms + image upload)

---

## 📂 Dataset
- Original dataset: [Tuberculosis Chest X-ray Dataset (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- Preprocessed + labeled dataset (ready-to-use splits): [Google Drive Link](https://drive.google.com/drive/folders/1MpTOFgSre2V4ueVPq1yQX2gJ0zedM1_e?usp=sharing)

⚠️ **Note**: Please credit the Kaggle dataset if you use this project.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, PyTorch
- **Frontend**: Next.js v15.5.2, Tailwind CSS, Shadcn UI
- **Languages**: Python 3.12.10, Node v22.16.0

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TBNet.git
cd TBNet
```

### 2. Backend Setup (FastAPI + PyTorch)

#### Create a Virtual Environment

```bash
python -m venv venv
```
```bash
venv\Scripts\activate      # Windows
```
```bash
source venv/bin/activate   # Linux/Mac
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Required key packages:

* torch / torchvision
* fastapi
* uvicorn
* pillow
* numpy

#### Download Dataset

* Download from Google Drive (link above)
* Place the dataset folder at the same level as `convolution.py`

```bash
backend/
├── convolution.py
├── test_convo.py
├── classification.py
├── dataset/   <-- put dataset here

```

#### Train Your Model

```bash
python convolution.py
```

* Automatically splits train/test sets
* Saves best model as `best_tb_model.pth`

#### Run FastAPI backend

```bash
uvicorn main:app --reload
```

* Backend will run at `http://127.0.0.1:8000`

### 3. Frontend Setup (Next.js)

```bash
cd frontend
npm install
npm run dev
```

* Runs at `http://localhost:3000`
* Connects to backend at `http://127.0.0.1:8000`

---

## 🔮 Usage Flow

1. Train CNN (`convolution.py`) → best model saved
2. Start FastAPI backend (`uvicorn main:app --reload`)
3. Start Next.js frontend (`npm run dev`)
4. Upload **symptoms** (JSON array) or **X-ray image** via frontend → prediction displayed

---

## 📁 Project Structure

```bash
TBNet/
├── backend/
│   ├── dataset/            # Chest X-ray dataset (user downloads separately)
│   ├── best_tb_model.pth   # Saved best CNN model (after training)
│   ├── classification.py   # Symptom classifier (MLP)
│   ├── convolution.py      # Train CNN on chest X-rays
│   ├── main.py             # FastAPI backend
│   ├── requirements.txt
│   └── test_convo.py       # Test CNN with pretrained model
├── frontend                # Next.js frontend
│   ├── .node_modules/      # will appear after initializing the Next.js Project
│   ├── public/
│   ├── src/
│   ├── other config files...
├── .gitignore
└── REAMDE.md

```

---

## 🙌 Credits

* Dataset: [Tuberculosis Chest X-ray Dataset (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* Libraries: PyTorch, FastAPI, Next.js, Tailwind, Shadcn UI

---

## 📌 Future Improvements
* Deploy trained model on cloud (AWS / GCP / HuggingFace Spaces)
* Extend dataset for better generalization
* Improve UI/UX for clinical usability

---

📜 License

MIT License – feel free to fork, modify, and use with attribution.
