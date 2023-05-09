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


def get_set_info(set_id, user_hash):
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


def get_price_guide(set_id):
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
    response = response.json()
    try:
        details = response["data"]["price_detail"]
        avg_price = response["data"]["avg_price"]
        return avg_price
    except:
        print(f"Uh oh: {response}")
        return 0



def main():
    """Main function for simple tests"""

    # # Just gets some space themes for now, price may not actually be there
    # user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    # death_star_price = get_set_info("75159-1", user_hash)
    # print(f"Death Star price: {death_star_price}")
    #
    # # Get price guide for a set
    # set_id = "75159-1"  # Death Star
    # price_guide = get_price_guide(set_id)
    # print(price_guide)

    # Build up dataset of features and price changes
    # Idea: use sets.csv as starting point - has set id and names and some other basic features
    # Then use name to get list price from lego_sets.csv
    # Then use set id to get price guide from Bricklink API
    # If we want more features we can look to Brickset or Bricklink API
    base = pd.read_csv("../data/sets.csv")
    list_price_df = pd.read_csv("../data/lego_sets.csv")
    list_price_df["prod_id"] = list_price_df["prod_id"].astype(int).astype(str)

    base["current_price"] = 0  # Beware of 0's as NA value
    base["list_price"] = 0
    for i in range(len(base)):
        set_id = base["set_num"][i]
        # list_price = list_price_df.loc[list_price_df['prod_id'] == set_id[:-2], 'list_price']
        # if not list_price.empty:
        #     base.loc[i, "current_price"] = get_price_guide(set_id)
        #     base.loc[i, "list_price"] = list_price
        base.loc[i, "current_price"] = get_price_guide(set_id)

    base.to_csv("custom.csv", index=False)


if __name__ == "__main__":
    main()


# Use aggregate data to create market index
# Day by day of aggregate market stuff
# Less data with trees
# Group it by theme and use as market sector
# Don't have to trade, could always analyze something else with book value or something like that
