import pytest
from fastapi.testclient import TestClient
from app import app  # Import your FastAPI app
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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
