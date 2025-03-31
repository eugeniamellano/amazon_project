import pytest
import json
import pandas as pd
from consumer import process_message  # Make sure to import the function correctly

# ------------------------------
# TEST 1: Process a valid message
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

# ------------------------------
# TEST 2: Message with no reviews
# ------------------------------
'''def test_process_message_no_reviews():
    test_message = json.dumps({
        "sellerId": "67890",
        "sellerName": "Gadget Hub",
        "reviews": []
    })

    df = process_message(test_message)
    
    # The function should return None if there are no reviews
    assert df is None

# ------------------------------
# TEST 3: Invalid message format
# ------------------------------
def test_process_invalid_message():
    test_message = "{invalid_json: true}"  # Malformed JSON
    
    df = process_message(test_message)
    
    # The function should handle the exception and return None
    assert df is None'''
