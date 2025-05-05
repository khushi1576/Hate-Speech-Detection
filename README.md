# Hate Speech Detection Classifier 🚨

This project uses deep learning and NLP techniques to classify tweets into three categories: 
- **Hate Speech**
- **Offensive Language**
- **Neither**

The classifier is built using an LSTM model and includes a user-friendly GUI application built with `Tkinter` and `ttkbootstrap` for easy interaction. The app classifies the tweets based on the trained model and displays the classification results.

## 🛠 Features

- **Tweet Classification:** Classifies tweets into three categories: "Hate Speech", "Offensive Language", or "Neither".
- **GUI Interface:** A modern and easy-to-use interface built using `Tkinter` and `ttkbootstrap` for seamless interaction.
- **Text Preprocessing:** Includes text normalization, punctuation removal, stopword filtering, and lemmatization using NLTK.
- **Model Training Script:** The LSTM model is trained on the dataset and saved for future use.
- **Pre-trained Model:** The project comes with a pre-trained model ready for deployment.
- **Tokenizer:** A saved tokenizer is included for easy preprocessing of input text.

## 📁 File Structure

```plaintext
Hate-Speech-Detection/
│
├── hate_speech.csv             # Dataset for training the model
├── hatespeechmodel.py          # Model training script
├── HateSpeechDetection.ipynb   # Jupyter notebook for training the model
├── gul.py                      # GUI script built with Tkinter
├── tokenizer.pkl               # Tokenizer for input text preprocessing       
├── requirements.txt            # List of required libraries
└── README.md                   # Project documentation
