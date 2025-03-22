from fastapi import FastAPI
import numpy as np
import pandas as pd
from typing import List
import re
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import mlflow.pyfunc


# Descargar stopwords si no están disponibles
nltk.download('stopwords')

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo guardado
#model = load_model('./amazon_model.h5')


# Cargar el dataset
df = pd.read_csv("Amazon_Unlocked_Mobile.csv")  #agregue esto

# Eliminar filas con valores nulos en la columna de reseñas
df.dropna(subset=["Reviews"], inplace=True)  #agregue esto


# Load MLflow model as a PyFuncModel
logged_model = 'runs:/633c81240a814a3aa3ec7de19a2e15df/lstm_model'
model = mlflow.pyfunc.load_model(logged_model)


# Inicializar el Stemmer y las stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Función de preprocesamiento
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = cleaned_text.lower().split()
    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# Función para crear y ajustar el tokenizador
def create_tokenizer(corpus):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(corpus)
    return tokenizer

# Endpoint para predecir el sentimiento
@app.post("/predict/")
def predict_sentiment(review: str):
    # Preprocesar el texto de la reseña
    processed_review = preprocess_text(review)
    
    # Crear el corpus a partir de la reseña procesada (si tienes más reseñas previas, puedes incluirlas aquí)
    corpus = [processed_review]
    
    # Crear y ajustar el tokenizador con el corpus
    tokenizer = create_tokenizer(corpus)
    
    # Convertir el texto procesado a secuencias numéricas
    sequence = tokenizer.texts_to_sequences([processed_review])
    
    # Rellenar las secuencias para que todas tengan la misma longitud
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    # Realizar la predicción utilizando el modelo cargado
    prediction = model.predict(padded_sequence)
    
    # Obtener la clase de sentimiento con la predicción
    sentiment_class = np.argmax(prediction)
    
    # Etiquetas de sentimiento
    sentiment_labels = {0: 'Negativo', 1: 'Neutral', 2: 'Positivo'}
    
    return {"sentimiento": sentiment_labels[sentiment_class]}


# Nuevo endpoint para previsualizar el dataset
@app.get("/preview/{number_row}", tags=["Data-Preview"], response_model=List[dict])  #agregue esto
async def preview_dataset(number_row: int = 5):  #agregue esto
    """
    Display a sample of rows of the dataset.
    `number_row` parameter allows to specify the number of rows you would like to display (default value: 5).
    """
    return df.head(number_row).to_dict(orient="records")  #agregue esto

# Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "API de análisis de sentimiento funcionando"}