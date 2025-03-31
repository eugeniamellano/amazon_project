FROM apache/airflow:2.10.4-python3.10

USER root  # Passer en root pour installer les dépendances

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow  # Revenir à l’utilisateur airflow pour la sécurité

# Copier les dépendances Python
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade --no-build-isolation -r requirements.txt