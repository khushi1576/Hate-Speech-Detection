# Hate Speech Detection

This project is a Hate Speech Classifier using Deep Learning and Natural Language Processing (NLP). It classifies tweets into three categories:

* Hate Speech
* Offensive Language
* Neither

The project includes a trained LSTM model, a tokenizer, and a user-friendly GUI using Tkinter and ttkbootstrap.

---

## 🚀 Features

* **Text Classification:** Classifies tweets into three categories.
* **GUI Interface:** Simple interface to input and predict in real-time.
* **Tokenizer & Pretrained Model:** Model and tokenizer saved for fast loading.

---

## 🗂 File Structure

```
HateSpeechDetection/
│
├── HateSpeechDetection.ipynb   # Jupyter notebook for training/testing
├── hatespeechmodel.py          # Script to train/save the model
├── gul.py                      # GUI using Tkinter + ttkbootstrap
├── hate_speech.csv             # Training dataset
├── tokenizer.pkl               # Saved tokenizer for text preprocessing
├── hate_speech_model.h5        # Trained LSTM model file
├── requirements.txt            # Dependencies list
├── LICENSE                     # MIT License file
└── README.md                   # Project documentation (this file)
```

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HateSpeechDetection.git
cd HateSpeechDetection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Train the Model (Optional if already trained)

```bash
python hatespeechmodel.py
```

### 2. Run the GUI App

```bash
python gul.py
```

Use the input field to enter a tweet, then click the **"Classify"** button to view the result.

---

## 📦 Requirements

* Python 3.x
* pandas
* numpy
* keras
* tensorflow
* sklearn
* nltk
* ttkbootstrap
* tkinter

Install via:

```bash
pip install -r requirements.txt
```

---

## 🧠 Dataset

The dataset (`hate_speech.csv`) contains tweets labeled as:

* **0**: Hate Speech
* **1**: Offensive Language
* **2**: Neither

It is used to train the LSTM model using Keras.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* [NLTK](https://www.nltk.org/)
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap)
* Dataset inspired by hate speech detection challenges

---

Made with ❤️ by \[Your Name]
