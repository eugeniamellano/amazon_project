import pytest
from fastapi.testclient import TestClient
from api import app  # Import your FastAPI app
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pytest
import json
import pandas as pd
from consumer import process_message  # Make sure to import the function correctly

# Initialize the FastAPI test client
client = TestClient(app)

# ------------------------------
# TEST 1: Prediction endpoint
# ------------------------------
def test_predict_sentiment():
    # Send a POST request to the /predict/ endpoint
    response = client.post(
        "/predict/",
        params={"review": "This phone is awesome!"}  # Send the input parameter in the query params
    )
    
    # Print the JSON response for debugging purposes
    print("Prediction Response:", response.json())

    # Verify that the response returns HTTP 200 OK
    assert response.status_code == 200

    # Parse the JSON response
    content = response.json()

    # Check if 'sentimiento' is present in the response
    assert "sentimiento" in content

    # Validate that the value is one of the expected classes
    assert content["sentimiento"] in ["Positivo", "Neutral", "Negativo"]


# ------------------------------
# TEST 2: Dataset preview endpoint
# ------------------------------
def test_preview_dataset():
    # Send a GET request to the /preview/ endpoint with a parameter
    response = client.get("/preview/3")

    # Print the JSON response for debugging purposes
    print("Preview Response:", response.json())

    # Verify that the response returns HTTP 200 OK
    assert response.status_code == 200

    # Parse the JSON response
    content = response.json()

    # Check that the response is a list
    assert isinstance(content, list)

    # Validate that the list has exactly 3 rows
    assert len(content) == 3

#KAFKA
# ------------------------------
# TEST 3: Process a valid message
# ------------------------------
def test_process_valid_message():
    # Sample test message in JSON format
    test_message = json.dumps({
        "sellerId": "12345",
        "sellerName": "Tech Store",
        "reviews": [
            {
                "reviewText": "Amazing product!",
                "rating": 5,
                "username": "user1",
                "date": "2024-03-24"
            },
            {
                "reviewText": "Not bad, but expected more.",
                "rating": 3,
                "username": "user2",
                "date": "2024-03-22"
            }
        ]
    })

    # Call the function
    df = process_message(test_message)

    # Ensure the output is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check that the expected columns are present
    expected_columns = {"reviewText", "rating", "username", "reviewDate", "sellerId", "sellerName"}
    assert set(df.columns) == expected_columns

    # Verify the processed data
    assert df.iloc[0]["reviewText"] == "Amazing product!"
    assert df.iloc[0]["rating"] == 5
    assert df.iloc[0]["username"] == "user1"
    assert df.iloc[0]["reviewDate"] == "2024-03-24"
    assert df.iloc[0]["sellerId"] == "12345"
    assert df.iloc[0]["sellerName"] == "Tech Store"

