# Utiliser l'image officielle Python
FROM python:3.9-slim

# Définir le dossier de travail
WORKDIR /app
RUN mkdir -p api

# Copier uniquement les fichiers nécessaires
COPY test/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copier le code source et les tests (avec le bon chemin)
COPY kafka/consumer/consumer_bis.py /app/
COPY app/app.py /app/api
COPY test/test.py test/upload_s3.py /app/

# Exécuter les tests Pytest au démarrage
# CMD ["pytest", "test_etl.py", "--junitxml=results.xml"]
CMD pytest test.py --junitxml=results.xml && python upload_s3.py
