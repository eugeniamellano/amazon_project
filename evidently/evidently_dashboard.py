import requests
import pickle
import pandas as pd
import numpy as np
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from io import BytesIO
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, ClassificationQualityByClass
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.report import Report
from evidently.test_suite import TestSuite


import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# URLs 
TFIDF_URL = "https://amazon-eugenia.s3.amazonaws.com/mlflow-lead/tfidf_vectorizer.pkl"
LOG_MODEL_URL = "https://amazon-eugenia.s3.amazonaws.com/mlflow-lead/logistic_regression_model.pkl"

# Download model and tokenizer
tfidf_vectorizer = pickle.load(BytesIO(requests.get(TFIDF_URL).content))
logistic_model = pickle.load(BytesIO(requests.get(LOG_MODEL_URL).content))

# Download csv
df_reference = pd.read_csv("https://amazon-eugenia.s3.us-east-1.amazonaws.com/mlflow-lead/result_data.csv")
df_production = pd.read_csv("https://amazon-eugenia.s3.us-east-1.amazonaws.com/mlflow-lead/production_data.csv")

# Preprocessing
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    words = cleaned_text.lower().split()  
    processed_words = [word for word in words if word not in STOP_WORDS]  # Delete stopwords
    return ' '.join(processed_words)

df_production['Processed_Reviews'] = df_production['Reviews'].apply(preprocess_text)

# Vectorizer
X_prod_tfidf = tfidf_vectorizer.transform(df_production['Processed_Reviews'])

# Predictions
y_pred_prod = logistic_model.predict(X_prod_tfidf)
df_production['Prediction'] = y_pred_prod

df_production['dataset'] = 'current'
df_reference['dataset'] = 'reference'

# Combine
df_combined = pd.concat([df_reference, df_production], ignore_index=True)

# Mapeping columns
column_mapping = ColumnMapping(
    target='Sentiment',       # Columna real en df_reference
    prediction='Prediction',  # Columna de predicción en df_production
    numerical_features=[],
    categorical_features=[]
)

# Report Evidently
report = Report(metrics=[
    DataDriftTable(),
    ClassificationQualityByClass()
])

# Report run
report.run(
    reference_data=df_reference,
    current_data=df_production,
    column_mapping=column_mapping
)



# Function to send email
def send_email(predictions):
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("EMAIL_USER")  # testing
    password = os.getenv("EMAIL_PASSWORD")

    subject = "Drift Alert"
    body = f"Alert: {predictions}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


report_dict= report.as_dict()

sentiment_drift_score = report.as_dict()["metrics"][0]["result"]["drift_by_columns"]["Sentiment"]["drift_score"]
prediction_drift_score = report.as_dict()["metrics"][0]["result"]["drift_by_columns"]["Prediction"]["drift_score"]

# check and define drift alert
if sentiment_drift_score > 0.5 or prediction_drift_score > 0.5:
    send_email(f"Sentiment Drift Score: {sentiment_drift_score}\nPrediction Drift Score: {prediction_drift_score}")




# save report as HTML
report.save_html('evidently_amazon.html')

print("Report Evidently saved as 'evidently_amazon.html'")
