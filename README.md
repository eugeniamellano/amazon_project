# Amazon Reviews Project - Sentiment Analysis

## Overview

This project automates the lifecycle of a machine learning model for sentiment analysis of Amazon reviews. The architecture integrates deployment, monitoring, and automated retraining using a variety of tools.

## Features

### Continuous Data Ingestion
Fetches reviews dynamically from an API via [RapidAPI](https://rapidapi.com/).

### Deployment
Deploys a FastAPI service to serve the model predictions.

### CI/CD Integration
Automates testing and deployment with Jenkins.

### Model and Monitoring
Uses MLflow for model versioning and Evidently for performance monitoring.

### Containerization
Dockerized components for easy deployment and scalability.

## Tech Stack

### Machine Learning
Scikit learn - keras

### API & Deployment
FastAPI, Docker

### CI/CD & Automation
Jenkins, Apache Airflow

### Monitoring & Model Management
MLflow, Evidently

### Data Pipeline
Data ingestion via RapidAPI

## Architecture
1. Fetches Amazon reviews dynamically from RapidAPI.
2. Processes and cleans the data for model training. ---> amazon_reviews.ipynb (EDA) , Model_amazon_project.ipynb (ML)
3. Automates ETL via Apache Airflow. ---> amazon_dag.py
4. Logs models using MLflow for versioning and tracking. ---> mlflow folder
5. Deploys the trained model as an API using FastAPI. -- folder app
6. Monitors model drift and performance using Evidently. ---> evidently folder
7. Automates test process with Jenkins. ---> jenkins folder
8. Hosts MLflow and FastAPI on Hugging Face Spaces for cloud accessibility

[FastAPI API](https://eugeniam-hfapi.hf.space/docs#)

[MLflow Tracking Server](https://eugeniam-hfmlflow.hf.space/)