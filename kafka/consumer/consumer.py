import os
import pandas as pd
from confluent_kafka import Consumer
import json
import boto3
import io
from dotenv import load_dotenv
import ccloud_lib

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET = os.getenv('S3_BUCKET')


# start config from "python.config"
CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC_INPUT = "amazon_topic"  # Topic for incoming data
TOPIC_OUTPUT = "amazon_topic"  # Topic for prediction output

# Create Consumer instance
consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
consumer_conf["group.id"] = "amazon_group"
consumer_conf["auto.offset.reset"] = "earliest"
consumer = Consumer(consumer_conf)

consumer.subscribe([TOPIC_INPUT])
print(f"Subscribed to topic: {TOPIC_INPUT}")

# Function to save CSV to S3
def save_to_s3(df, bucket_name, aws_access_key, aws_secret_key, region, object_name=None):
    if object_name is None:
        # Create a unique object name based on the current timestamp
        object_name = f"reviews/reviews_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Create a buffer and save the DataFrame as CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Initialize the S3 client
    s3_client = boto3.client('s3', 
                             aws_access_key_id=aws_access_key, 
                             aws_secret_access_key=aws_secret_key, 
                             region_name=region)

    # Upload the CSV to S3
    s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=csv_buffer.getvalue())
    print(f"CSV file saved to S3 as {object_name}")

# Function to process each Kafka message
def process_message(msg_value):
    try:
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

    except Exception as e:
        print(f"Error processing message: {e}")
        return None

print("Receiving messages from Kafka...")

# Buffer to accumulate messages (5 records)
df_accumulated = pd.DataFrame()

try:
    while True:
        msg = consumer.poll(1.0)  # Wait for a message

        if msg is None:
            continue  # No message, continue polling
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        print(f"\nMessage received: {msg.value().decode('utf-8')}")

        # Process the message and convert it to a DataFrame
        df = process_message(msg.value().decode('utf-8'))

        if df is not None:
            # Add the processed DataFrame to the accumulated buffer
            df_accumulated = pd.concat([df_accumulated, df], ignore_index=True)
            print(f"Added {len(df)} records to the batch. Current batch size: {len(df_accumulated)} records.")

        # Save and reset buffer when we reach 5 records
        if len(df_accumulated) >= 5:
            print(f"Saving {len(df_accumulated)} records to S3...")
            save_to_s3(df_accumulated, S3_BUCKET, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
            print("Data saved to S3, resetting buffer.")
            df_accumulated = pd.DataFrame()  # Reset the buffer for new data

except KeyboardInterrupt:
    print("Interrupted. Closing consumer...")

finally:
    consumer.close()
