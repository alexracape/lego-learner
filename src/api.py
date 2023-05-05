import requests
from requests_oauthlib import OAuth1
from dotenv import load_dotenv
import os
import json
import pandas as pd

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path)

# Bricklink API Credentials
BL_USERNAME = os.getenv('BL_USERNAME')
BL_PASSWORD = os.getenv('BL_PASSWORD')
BL_API_TOKEN = os.getenv('BL_API_TOKEN')
BL_API_SECRET = os.getenv('BL_API_SECRET')
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')

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
    api_endpoint = f"https://api.bricklink.com/api/store/v1/items/SET/{set_id}/price"
    oauth = OAuth1(
        client_key=CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=BL_API_TOKEN,
        resource_owner_secret=BL_API_SECRET,
    )
    params = {
        "guide_type": "sold",  # sold (closed) or stock (active listings)
        "new_or_used": "N",
        "currency_code": "USD",
        "region": "north_america",
    }

    # Send the API request
    response = requests.get(api_endpoint, auth=oauth, params=params)

    # Return the price
    return response.json()


def main():
    """Main function for simple tests"""

    # Just gets some space themes for now, price may not actually be there
    user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    death_star_price = get_set_price("75159-1", user_hash)
    print(f"Death Star price: {death_star_price}")

    # Get price guide for a set
    set_id = "75159-1"  # Death Star
    price_guide = get_set_historical(set_id)
    print(price_guide)


if __name__ == "__main__":
    main()


# Use aggregate data to create market index
# Day by day of aggregate market stuff
# Less data with trees
# Group it by theme and use as market sector
# Don't have to trade, could always analyze something else with book value or something like that
