import os
import pandas as pd
from confluent_kafka import Consumer
import json
import boto3
import io
from dotenv import load_dotenv
import ccloud_lib


# Function to process each Kafka message
def process_message(msg_value):
    
    # Parse the message into a JSON object
    data = json.loads(msg_value)

    # Extract relevant data
    seller_id = data.get("sellerId")
    seller_name = data.get("sellerName")
    reviews = data.get("reviews", [])

    if not reviews:
        print("No reviews in this message.")
        return None

    # Create a DataFrame with relevant fields
    df = pd.DataFrame(reviews)
    df = df[['reviewText', 'rating', 'username', 'date']].copy()
    df['rating'] = df['rating'].astype(int)

    # Add seller-related fields
    df['sellerId'] = seller_id
    df['sellerName'] = seller_name

    # Rename the 'date' column to avoid conflicts
    df.rename(columns={'date': 'reviewDate'}, inplace=True)

    return df
