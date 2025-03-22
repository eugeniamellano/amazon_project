import mlflow
import mlflow.keras
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score # #Added import for additional metrics
import pickle

# Configure the URI for the MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # Make sure the server is running on this port

# Load data
df = pd.read_csv('Amazon_Unlocked_Mobile.csv').dropna(subset=['Reviews', 'Rating'])
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorization for Logistic Regression
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Tokenization for LSTM
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

# Start the first experiment for Logistic Regression
mlflow.set_experiment("regression_experiment")  # #Change: Set experiment for Logistic Regression
with mlflow.start_run():
    # Logistic Regression Model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_tfidf, y_train)
    acc_log = log_model.score(X_test_tfidf, y_test)
    
    # Log Logistic Regression
    mlflow.sklearn.log_model(log_model, "logistic_regression_model")
    mlflow.log_metric("log_reg_accuracy", acc_log)
    
    # Save and log the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
        mlflow.log_artifact("tfidf_vectorizer.pkl")
    
    # Additional metrics for Logistic Regression
    y_pred_log = log_model.predict(X_test_tfidf)  # #Added: Predictions for Logistic Regression
    mlflow.log_metric("log_reg_precision", accuracy_score(y_test, y_pred_log))  # #Added: Precision
    mlflow.log_metric("log_reg_recall", accuracy_score(y_test, y_pred_log))  # #Added: Recall
    mlflow.log_metric("log_reg_f1", accuracy_score(y_test, y_pred_log))  # #Added: F1-Score

    print(f"Logistic Regression - Accuracy: {acc_log:.2f}")

# Start the second experiment for LSTM
mlflow.set_experiment("lstm_experiment")  # #Change: Set experiment for LSTM
with mlflow.start_run():
    # LSTM Model
    lstm_model = Sequential([
        Embedding(5000, 128, input_length=200),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(3, activation='softmax')
    ])
    
    lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), epochs=5, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    
    loss, acc_lstm = lstm_model.evaluate(X_test_seq, y_test)
    
    # Log LSTM model
    mlflow.keras.log_model(lstm_model, "lstm_model")
    mlflow.log_metric("lstm_accuracy", acc_lstm)
    
    # Save and log the Tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
        mlflow.log_artifact("tokenizer.pkl")
    
    # Additional metrics for LSTM
    y_pred_lstm = np.argmax(lstm_model.predict(X_test_seq), axis=1)  # #Added: Predictions for LSTM
    mlflow.log_metric("lstm_precision", accuracy_score(y_test, y_pred_lstm))  # #Added: Precision
    mlflow.log_metric("lstm_recall", accuracy_score(y_test, y_pred_lstm))  # #Added: Recall
    mlflow.log_metric("lstm_f1", accuracy_score(y_test, y_pred_lstm))  # #Added: F1-Score


    print(f"LSTM - Accuracy: {acc_lstm:.2f}")

    print(f"LSTM - Accuracy: {acc_lstm:.2f}")