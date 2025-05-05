# 🛡️ Hate Speech Detection

A Deep Learning and NLP-based project that classifies tweets into:

- **Hate Speech**
- **Offensive Language**
- **Neither**

This project includes both model training and a GUI for live tweet classification.

---

## 📁 Project Structure

| File / Folder        | Description                                       |
|----------------------|---------------------------------------------------|
| `hate_speech.csv`    | Dataset used for training                         |
| `HateSpeechDetection.ipynb` | Jupyter notebook for training & testing the model |
| `hatespeechmodel.py` | Python script to train and save the model         |
| `tokenizer.pkl`      | Tokenizer saved for preprocessing input tweets    |
| `gul.py`             | GUI app using `tkinter` and `ttkbootstrap`        |

---

## 🧠 Model Details

- **Embedding + Bidirectional LSTM** architecture
- Categorical cross-entropy loss
- Trained using balanced tweet data
- Prediction classes:
  - 0 → Hate Speech
  - 1 → Offensive Language
  - 2 → Neither

---

## 🖥️ GUI App

A desktop interface built using `ttkbootstrap`, allowing you to input any tweet and instantly classify it.

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
