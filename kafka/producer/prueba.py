import requests

API_URL = "https://amazon-product-reviews-keywords.p.rapidapi.com/product/reviews"

querystring = {
    "asin": "B07XQXZXJC",
    "country": "US",
    "variants": "1",
    "top": "0"
}

headers = {
    "x-rapidapi-key": "6437ccf1ffmsh146bd4283c2592ep11cc72jsna5e4a3d77eb1",
    "x-rapidapi-host": "amazon-product-reviews-keywords.p.rapidapi.com"
}

response = requests.get(API_URL, headers=headers, params=querystring)
print(response.status_code)
print(response.text)