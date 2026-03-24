# EX NO: Sentiment Analysis using LSTM

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# STEP 1: Create Dataset
# -------------------------------
data = {
    "text": [
        # Positive
        "I love this product", "This is amazing", "Very happy with the service",
        "Absolutely fantastic", "I enjoyed this a lot", "Superb experience",
        "Excellent product", "Really good", "Loved it",

        # Negative
        "I hate this", "Worst experience ever", "Not good at all",
        "Very bad product", "I am disappointed", "Terrible service",
        "This is terrible", "Awful experience", "Horrible product",

        # Neutral
        "It is okay", "Average product", "Nothing special",
        "It is fine", "Not bad not good", "Just okay",
        "Moderate quality", "So so experience", "Fine product"
    ],
    "sentiment": [
        "positive","positive","positive","positive","positive","positive","positive","positive","positive",
        "negative","negative","negative","negative","negative","negative","negative","negative","negative",
        "neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral"
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# STEP 2: Convert Labels
# -------------------------------
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(label_map)

# -------------------------------
# STEP 3: Tokenization
# -------------------------------
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])

# -------------------------------
# STEP 4: Padding
# -------------------------------
max_len = 10
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df["sentiment"]

# -------------------------------
# STEP 5: USE FULL DATA (IMPORTANT FIX)
# -------------------------------
X_train, X_test = X, X
y_train, y_test = y, y

# -------------------------------
# STEP 6: Build LSTM Model
# -------------------------------
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# -------------------------------
# STEP 7: Compile Model
# -------------------------------
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------------------------
# STEP 8: Train Model
# -------------------------------
model.fit(X_train, y_train, epochs=50, verbose=1)

# -------------------------------
# STEP 9: Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nModel Accuracy:", round(accuracy, 2))

# -------------------------------
# STEP 10: Prediction Function
# -------------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)
    label = np.argmax(pred)

    labels = ["Negative", "Neutral", "Positive"]
    return labels[label]

# -------------------------------
# STEP 11: Test Predictions
# -------------------------------
print("\nPredictions:")
print("I really love this ->", predict_sentiment("I really love this"))
print("This is terrible ->", predict_sentiment("This is terrible"))
print("It is fine ->", predict_sentiment("It is fine"))