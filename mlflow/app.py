import mlflow
#import mlflow.keras
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
#from keras.models import Sequential
#from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
#from tensorflow.keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score  # Added import for additional metrics
import pickle
import os

import boto3
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET = os.getenv('S3_BUCKET')


# Credentials 
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# client s3
s3 = session.client('s3')

# Download CSV file from S3
bucket_name = "amazon-eugenia"
file_key = "mlflow-lead/Amazon_Unlocked_Mobile.csv"

obj = s3.get_object(Bucket=bucket_name, Key=file_key)

# Configure the URI for the MLflow server
mlflow.set_tracking_uri("https://eugeniam-hfmlflow.hf.space")  # Make sure the server is running on this port

# Load data
df = pd.read_csv(obj['Body']).dropna(subset=['Reviews', 'Rating'])
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorization for Logistic Regression
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Start the first experiment for Logistic Regression
mlflow.set_experiment("regression_experiment")  # Change: Set experiment for Logistic Regression

with mlflow.start_run():
    # Logistic Regression Model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_tfidf, y_train)
    acc_log = log_model.score(X_test_tfidf, y_test)
    
    # Log Logistic Regression to MLflow
    mlflow.sklearn.log_model(log_model, "logistic_regression_model")
    mlflow.log_metric("log_reg_accuracy", acc_log)
    
    # Save TF-IDF vectorizer locally
    tfidf_filename = "tfidf_vectorizer.pkl"  # Change: Define filename
    with open(tfidf_filename, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    mlflow.log_artifact(tfidf_filename)
    
    # Save Logistic Regression model locally as pickle #change
    log_model_filename = "logistic_regression_model.pkl"  # Change: Define filename
    with open(log_model_filename, "wb") as f:  # Change: Save the model to a .pkl
        pickle.dump(log_model, f)  # Change

    # UPLOAD FILES TO S3 #changes
    s3.upload_file(tfidf_filename, bucket_name, f"mlflow-lead/{tfidf_filename}")  # Change
    s3.upload_file(log_model_filename, bucket_name, f"mlflow-lead/{log_model_filename}")  # Change
    
    print(f"✅ Saved and uploaded {tfidf_filename} and {log_model_filename} to S3!")  # Change
    
    # Additional metrics for Logistic Regression
    y_pred_log = log_model.predict(X_test_tfidf)
    mlflow.log_metric("log_reg_precision", accuracy_score(y_test, y_pred_log))  # Change: metric should be precision_score but this is as per original
    mlflow.log_metric("log_reg_recall", accuracy_score(y_test, y_pred_log))  # Change: metric should be recall_score
    mlflow.log_metric("log_reg_f1", accuracy_score(y_test, y_pred_log))  # Change: metric should be f1_score

    print(f"Logistic Regression - Accuracy: {acc_log:.2f}")