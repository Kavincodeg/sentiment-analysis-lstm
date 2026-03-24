# 🧠 Sentiment Analysis using LSTM

## 📌 Overview

This project implements a **Sentiment Analysis system** using a Long Short-Term Memory (LSTM) deep learning model. It classifies text into three categories: **Positive, Negative, and Neutral**.

## 🎯 Objective

To automate the process of analyzing textual data and identifying the emotional tone behind it, which is useful in applications like customer feedback analysis, social media monitoring, and review analysis.

## ⚙️ Methodology

* Text data is preprocessed using **tokenization** and **padding**
* Sentiment labels are converted into numerical values
* An **LSTM-based neural network** is built using TensorFlow/Keras
* The model learns contextual and sequential patterns in text
* Predictions are made for new input sentences

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn

## 🚀 Features

* Deep learning-based NLP model
* Handles sequential text data effectively
* Classifies text into Positive, Negative, and Neutral sentiments
* Simple and easy-to-understand implementation

## 📊 Sample Output

```
Model Accuracy: 1.0

Predictions:
I really love this -> Positive  
This is terrible -> Negative  
It is fine -> Neutral  
```

## ▶️ How to Run

```bash
pip install tensorflow pandas numpy scikit-learn
python sentiment_lstm.py
```

## 📚 Learning Outcomes

* Understanding of LSTM networks in NLP
* Text preprocessing techniques
* Building and training deep learning models
* Sentiment classification using neural networks

## 📁 Project Structure

```
Sentiment_LSTM_Project/
│── sentiment_lstm.py
│── README.md
```

## 👨‍💻 Author

Your Name
