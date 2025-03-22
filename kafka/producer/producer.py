import requests
from confluent_kafka import Producer
import json
import ccloud_lib
import time

CONF = ccloud_lib.read_ccloud_config("python.config")
TOPIC = "amazon_topic"

producer_conf = ccloud_lib.pop_schema_registry_params_from_config(CONF)
producer = Producer(producer_conf)

API_URL = "https://realtime-amazon-data.p.rapidapi.com/seller-details"

querystring = {
    "sellerId": "AXOGFIT0PZZ7G",  
    "country": "in"
}

headers = {
    "x-rapidapi-key": "6437ccf1ffmsh146bd4283c2592ep11cc72jsna5e4a3d77eb1",
    "x-rapidapi-host": "realtime-amazon-data.p.rapidapi.com"
}

# Delivery report callback to confirm successful message delivery
def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Error sending message: {err}")
    else:
        print(f"✅ Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
        print(f"📦 Message content: {msg.value().decode('utf-8')}")

# Function to fetch data from the API and produce messages to Kafka
def fetch_and_produce_reviews():
    while True:
        try:
            response = requests.get(API_URL, headers=headers, params=querystring)

            if response.status_code != 200:
                print(f"❌ API response error: {response.status_code}")
                time.sleep(10)
                continue

            data = response.json()

            # Send the complete data (adjust if you want to extract specific parts)
            review_json = json.dumps(data)

            producer.produce(
                topic=TOPIC,
                value=review_json,
                callback=delivery_report
            )

            producer.poll(0)
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error during execution: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    print("🚀 Starting to send reviews to Kafka...")
    fetch_and_produce_reviews()
    producer.flush()
