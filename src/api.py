import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
import json

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path)

# Bricklink API Credentials
BL_USERNAME = os.getenv('BL_USERNAME')
BL_PASSWORD = os.getenv('BL_PASSWORD')
BL_API_KEY = os.getenv('BL_API_KEY')
BL_API_SECRET = os.getenv('BL_API_SECRET')

# Brickset API Credentials
BS_API_KEY = os.environ.get('BS_API_KEY')
BS_USERNAME = os.environ.get('BS_USERNAME')
BS_PASSWORD = os.environ.get('BS_PASSWORD')


def get_set_price(set_id, user_hash):
    """Use Brickset API to get the price of a set"""

    # Set the API endpoint and parameters
    api_endpoint = "https://brickset.com/api/v3.asmx/getSets"
    params = {
        "apiKey": BS_API_KEY,
        "userHash": user_hash,
        "params": json.dumps({
            "theme": "space"
        })
    }

    # Send the API request
    response = requests.get(api_endpoint, params=params)

    # Return the price
    return response.json()
    # return response.json()["sets"][0]["retailPrice"]


def get_user_hash(username, password):
    """Use Brickset API to get the user hash"""

    # Set the API endpoint and parameters
    api_endpoint = "https://brickset.com/api/v3.asmx/login"
    params = {
        "apiKey": BS_API_KEY,
        "username": username,
        "password": password
    }

    # Send the API request
    response = requests.get(api_endpoint, params=params)

    # Return the price
    return response.json()["hash"]

def get_set_historical(set_id):
    """Use Bricklink API to get the historical price of a set"""

    # Set the API endpoint and parameters
    api_endpoint = "https://api.bricklink.com/api/store/v1/items/set/{set_id}/price".format(set_id=set_id)
    params = {
        "guide_type": "sold",  # sold (closed) or stock (active listings)
        "new_or_used": "N",
        "currency_code": "USD",
        "region": "US",
        "q": "price",
        "username": BL_USERNAME,
        "password": BL_PASSWORD,
        "consumer_key": BL_API_KEY,
        "consumer_secret": BL_API_SECRET,
    }
    auth = HTTPBasicAuth(BL_USERNAME, BL_PASSWORD)
    # Send the API request
    response = requests.get(api_endpoint, params=params)

    # Return the price
    return response.json()["data"][0]["unit_price"]


def main():
    """Main function for simple tests"""

    hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    death_star_price = get_set_price("75159-1", hash)
    print(f"Death Star price: {death_star_price}")


if __name__ == "__main__":
    main()
