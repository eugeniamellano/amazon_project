from fastapi import FastAPI
import numpy as np
import pandas as pd
from typing import List
import re
import pickle
import boto3
from spacy.lang.en.stop_words import STOP_WORDS
import os
from dotenv import load_dotenv
import requests
from io import BytesIO

# URLs públicas de los archivos
TFIDF_URL = "https://amazon-eugenia.s3.amazonaws.com/mlflow-lead/tfidf_vectorizer.pkl"
LOG_MODEL_URL = "https://amazon-eugenia.s3.amazonaws.com/mlflow-lead/logistic_regression_model.pkl"

# Descargar y cargar el vectorizador TF-IDF y el modelo de regresión logística
tfidf_vectorizer = pickle.load(BytesIO(requests.get(TFIDF_URL).content))
log_model = pickle.load(BytesIO(requests.get(LOG_MODEL_URL).content))
df=pd.read_csv("https://amazon-eugenia.s3.us-east-1.amazonaws.com/mlflow-lead/Amazon_Unlocked_Mobile.csv")


'''
# Cargar variables de entorno
load_dotenv()
#AWS_ACCESS_KEY = os.environ('AWS_ACCESS_KEY_ID')
#AWS_SECRET_KEY = os.environ('AWS_SECRET_ACCESS_KEY')
AWS_REGION = 'us-east-1'
S3_BUCKET = 'amazon-eugenia'
'''
# Inicializar FastAPI
app = FastAPI()


# Inicializar stopwords de spaCy
stop_words = STOP_WORDS

# Función de preprocesamiento de texto
def preprocess_text(text):
    """
    Preprocesa el texto de entrada:
    - Elimina caracteres no alfabéticos
    - Convierte todo a minúsculas
    - Elimina las stopwords
    """
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)  # Eliminar caracteres no alfabéticos
    words = cleaned_text.lower().split()  # Convertir a minúsculas y dividir en palabras
    processed_words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    return ' '.join(processed_words)

# Endpoint para predecir el sentimiento
@app.post("/predict/")
def predict_sentiment(review: str):
    """
    Predice el sentimiento de la reseña proporcionada utilizando el modelo de regresión logística preentrenado.
    """
    # Preprocesar la reseña de entrada
    processed_review = preprocess_text(review)
    
    # Vectorizar la reseña preprocesada usando el vectorizador TF-IDF cargado
    review_vectorized = tfidf_vectorizer.transform([processed_review])
    
    # Predecir la clase de sentimiento usando el modelo de regresión logística cargado
    prediction = log_model.predict(review_vectorized)[0]
    
    # Etiquetas de sentimiento
    sentiment_labels = {
        0: 'Negativo',
        1: 'Positivo',
        2: 'Neutral'
    }
    
    # Retornar la etiqueta de sentimiento predicha
    return {"sentiment": sentiment_labels.get(prediction, "Unknown")}

# Endpoint para previsualizar filas del dataset
@app.get("/preview/{number_row}", tags=["Data-Preview"], response_model=List[dict])
async def preview_dataset(number_row: int = 5):
    """
    Muestra una muestra de filas del dataset almacenado en S3.
    """
        
    # Eliminar filas con reseñas faltantes
    df.dropna(subset=["Reviews"], inplace=True)
    
    # Retornar las primeras 'number_row' filas como diccionarios
    return df.head(number_row).to_dict(orient="records")

# Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "API is working"}
