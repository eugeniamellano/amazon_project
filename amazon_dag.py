import json
import logging
from datetime import datetime

import pandas as pd
import requests
from airflow import DAG
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from s3_to_postgres import S3ToPostgresOperator

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

# -- API Config --
API_URL = "https://realtime-amazon-data.p.rapidapi.com/seller-details"
querystring = {"sellerId": "AXOGFIT0PZZ7G", "country": "in"}
headers = {
    "x-rapidapi-key": Variable.get("RapidAPIKey"),
    "x-rapidapi-host": "realtime-amazon-data.p.rapidapi.com",
}


# -- Task 1: Fetch + upload raw JSON to S3 --
def fetch_seller_reviews(**context):
    logging.info("🔍 Fetching seller reviews from API...")
    response = requests.get(API_URL, headers=headers, params=querystring)

    if response.status_code != 200:
        raise Exception(
            f"❌ API request failed with status code {response.status_code}"
        )

    data = response.json()
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_amazon_raw.json"
    local_path = f"/tmp/{filename}"

    with open(local_path, "w") as f:
        json.dump(data, f)

    # Upload to S3
    s3_hook = S3Hook(aws_conn_id="aws_default")
    s3_hook.load_file(
        filename=local_path,
        key=f"raw/{filename}",
        bucket_name=Variable.get("S3BucketName"),
    )

    context["ti"].xcom_push(key="raw_json_filename", value=filename)
    logging.info(f"✅ Uploaded raw JSON to S3 as raw/{filename}")


# -- Task 2: Transform reviews and upload CSV to S3 (only if ≥ 5) --
def transform_and_upload_csv(**context):
    filename = context["ti"].xcom_pull(key="raw_json_filename")
    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_path = s3_hook.download_file(
        key=f"raw/{filename}",
        bucket_name=Variable.get("S3BucketName"),
        local_path="/tmp",
    )

    with open(local_path, "r") as f:
        data = json.load(f)

    reviews = data.get("reviews", [])
    if not reviews or len(reviews) < 5:
        logging.info("⏸️ Not enough reviews to process. Skipping upload.")
        return

    df = pd.DataFrame(reviews)
    df = df[["reviewText", "rating", "username", "date"]].copy()
    df["rating"] = df["rating"].astype(int)
    df["sellerId"] = data.get("sellerId")
    df["sellerName"] = data.get("sellerName")
    df.rename(columns={"date": "reviewDate"}, inplace=True)

    csv_filename = filename.replace(".json", ".csv")
    csv_full_path = f"/tmp/{csv_filename}"
    df.to_csv(csv_full_path, index=False, header=False)

    s3_hook.load_file(
        filename=csv_full_path,
        key=f"reviews/{csv_filename}",
        bucket_name=Variable.get("S3BucketName"),
    )

    context["ti"].xcom_push(key="reviews_csv_filename", value=f"reviews/{csv_filename}")
    logging.info(f"✅ Uploaded transformed CSV to S3 as reviews/{csv_filename}")


# -- DAG Definition --
with DAG(
    dag_id="etl_amazon_reviews_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
    tags=["amazon", "etl", "s3", "postgres"],
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_seller_reviews",
        python_callable=fetch_seller_reviews,
        provide_context=True,
    )

    transform_and_save_task = PythonOperator(
        task_id="transform_and_upload_csv_if_enough_reviews",
        python_callable=transform_and_upload_csv,
        provide_context=True,
    )

    create_table_task = PostgresOperator(
        task_id="create_amazon_reviews_table",
        sql="""
            CREATE TABLE IF NOT EXISTS amazon_reviews (
                reviewText TEXT,
                rating INTEGER,
                username TEXT,
                reviewDate TEXT,
                sellerId TEXT,
                sellerName TEXT
            );
        """,
        postgres_conn_id="postgres_default",
    )

    load_to_postgres_task = S3ToPostgresOperator(
        task_id="s3_to_postgres_amazon_reviews",
        table="amazon_reviews",
        bucket="{{ var.value.S3BucketName }}",
        key="{{ task_instance.xcom_pull(task_ids='transform_and_upload_csv_if_enough_reviews', key='reviews_csv_filename') }}",
        postgres_conn_id="postgres_default",
        aws_conn_id="aws_default",
    )

    fetch_task >> transform_and_save_task >> create_table_task >> load_to_postgres_task
