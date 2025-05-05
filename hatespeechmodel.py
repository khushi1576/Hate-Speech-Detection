import pandas as pd
import numpy as np
import string
import nltk
import warnings
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import layers# type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer# type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau# type: ignore

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("hate_speech.csv")

# Lowercase all text
df['tweet'] = df['tweet'].str.lower()

# Remove punctuation
def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['tweet'] = df['tweet'].apply(remove_punctuations)

# Remove stopwords and lemmatize
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]
    return ' '.join(words)

df['tweet'] = df['tweet'].apply(preprocess_text)

# Balance the dataset
class_2 = df[df['class'] == 2]
class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)
class_0 = df[df['class'] == 0]

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2])

# Split into features and targets
X = balanced_df['tweet']
y = pd.get_dummies(balanced_df['class'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=22)

# Tokenization
max_words = 5000
max_len = 50
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

# Build model
model = Sequential([
    layers.Embedding(max_words, 32, input_length=max_len),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(512, activation='relu', kernel_regularizer='l1'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5)

# Train model
model.fit(X_train_pad, y_train,
          validation_data=(X_val_pad, y_val),
          epochs=10,
          batch_size=32,
          callbacks=[es, lr],
          verbose=1)

# Save model
model.save("hate_speech_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved!")
