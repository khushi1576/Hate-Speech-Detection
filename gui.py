import tkinter as tk
from tkinter import messagebox
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import pickle
import ttkbootstrap as ttk
from ttkbootstrap import Window

# Load model and tokenizer
model = load_model('hate_speech_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 50

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split() if w not in stop_words
    ]
    return ' '.join(words)

# GUI logic for classification
def classify_text():
    input_text = entry.get()
    if not input_text.strip():
        messagebox.showwarning("Input Error", "Please enter a tweet to classify.")
        return
    try:
        classify_button.configure(state='disabled', text="Classifying...")
        processed = preprocess_text(input_text)
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        prediction = model.predict(padded)[0]
        result = np.argmax(prediction)
        labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
        output_label.config(text=f"Prediction: {labels[result]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        classify_button.configure(state='normal', text="Classify")

# Clear input and output
def clear_input():
    entry.delete(0, tk.END)
    output_label.config(text="")

# Initialize ttkbootstrap window with dark theme
root = Window(themename='darkly', title="Hate Speech Detector", size=(500, 350), resizable=(False, False))

# Set up GUI components
ttk.Label(root, text="Enter a tweet:", font=("Helvetica", 14)).pack(pady=(20, 10))

entry = ttk.Entry(root, width=50, font=("Helvetica", 12))
entry.pack(pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=15)

classify_button = ttk.Button(button_frame, text="Classify", command=classify_text, style="success.TButton")
classify_button.grid(row=0, column=0, padx=10)

clear_button = ttk.Button(button_frame, text="Clear", command=clear_input, style="warning.TButton")
clear_button.grid(row=0, column=1, padx=10)

output_label = ttk.Label(root, text="", font=("Helvetica", 14, "bold"), bootstyle="info")
output_label.pack(pady=20)

root.mainloop()
