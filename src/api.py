import requests
from requests_oauthlib import OAuth1
from dotenv import load_dotenv
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def get_prices_by_year(year, df, user_hash):
    """Use Brickset API to get the price of a set"""

    # Set the API endpoint and parameters
    api_endpoint = "https://brickset.com/api/v3.asmx/getSets"
    params = {
        "apiKey": BS_API_KEY,
        "userHash": user_hash,
        "params": json.dumps({
            "year": year,
        })
    }

    # Send the API request
    response = requests.get(api_endpoint, params=params)
    response = response.json()
    if "message" in response:
        print("API Limit Exceeded")
        return True

    # Lots more features in here if we need it
    try:
        sets = response["sets"]
        for set in sets:
            set_number = set["number"]
            set_variant = set["numberVariant"]
            set_id = f"{set_number}-{set_variant}"
            try:
                historical_price = set["LEGOCom"]["US"]["retailPrice"]
                df.loc[set_id, "list_price"] = historical_price
            except Exception as e:
                print(f"Could not find historical price for {set_id} -- {e}")
                historical_price = 0
    except Exception as e:
        print(f"Could not find historical price for {set_id} -- {e}")
        historical_price = 0

    # Return the price
    return False


def get_historical_price(set_id, user_hash):
    """Use Brickset API to get the price of a set"""

    # Set the API endpoint and parameters
    api_endpoint = "https://brickset.com/api/v3.asmx/getSets"
    params = {
        "apiKey": BS_API_KEY,
        "userHash": user_hash,
        "params": json.dumps({
            "setNumber": set_id,
        })
    }

    # Send the API request
    response = requests.get(api_endpoint, params=params)
    response = response.json()
    # Lots more features in here if we need it
    try:
        historical_price = response["sets"][0]["LEGOCom"]["US"]["retailPrice"]
    except Exception as e:
        print(f"Could not find historical price for {set_id} -- {e}")
        historical_price = 0

    # Return the price
    return historical_price


def get_current_price(set_id):
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
        # Get most recent order for price
        orders = response["data"]["price_detail"]
        most_recent_date = datetime.min
        most_recent_order = None
        for order in orders:
            date = datetime.strptime(order["date_ordered"], "%Y-%m-%dT%H:%M:%S.%fZ")
            if date > most_recent_date:
                most_recent_date = date
                most_recent_order = order
        return most_recent_order["unit_price"]
    except Exception as e:
        print(f"Uh oh: {e} for {set_id}")
        return 0


def get_all_current_prices():
    """Get all current prices for all sets in the database"""

    base = pd.read_csv("../data/catalog.csv")
    list_price_df = pd.read_csv("../data/old_data_source.csv")
    list_price_df["prod_id"] = list_price_df["prod_id"].astype(int).astype(str)

    base["current_price"] = 0
    base["list_price"] = 0
    for i in range(len(base)):
        set_id = base["set_num"][i]
        current_price = get_current_price(set_id)
        base.loc[i, "current_price"] = current_price

    base.to_csv("custom_2.csv", index=False)


def get_all_list_prices(year):
    """Use Brickset to get list prices from lego.com

    This approach was abandoned, since the API was returning 0's for almost all sets before 2015 ish
    - got 788 prices
    - only 175 overlapped with current trades
    """
    user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    df = pd.read_csv("../data/custom_2.csv")
    df = df.set_index("set_num")
    for year in range(2000, 2024):
        at_limit = get_prices_by_year(year, df, user_hash)
        if at_limit:
            print(f"API LIMIT REACHED: stopped at {year}")
            break
        else:
            print(f"Finished {year}")
    df.to_csv("custom_3.csv", index=True)
    print(df)


def simple_tests():
    """Some basic tests to check API calls"""

    # Pick an arbitrary set
    set_id = "75159-1"  # Death Star

    # Get historical price from bricklink
    user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    death_star_price = get_historical_price(set_id, user_hash)
    print(f"Death Star price: {death_star_price}")

    # Get price guide for a set
    price_guide = get_current_price(set_id)
    print(price_guide)


def eda(data_source="custom_4.csv"):
    """Some basic EDA to explore the scraped sets / price data"""

    # Load data
    data = pd.read_csv(f"../data/{data_source}")

    # Check how much data has both list and current price
    with_both = data[(data["current_price"].notna()) & (data["USD_MSRP"].notna())]
    print(f"Data with both list and current price: {len(with_both)}")

    # Plot the price vs year and histogram of price
    data.plot(x="Year", y="current_price", kind="scatter", logy=True, xlabel="Year", ylabel="Current Price")
    data.plot(y="current_price", kind="hist", logy=True, bins=100)
    data.plot(x="Year", y="USD_MSRP", kind="scatter", logy=True, xlabel="Year", ylabel="List Price")
    plt.show()


def main():
    """Main function for manipulating csvs and moving data around"""

    # Load Data Sets
    custom = pd.read_csv("../data/custom_4.csv")
    lego_sets = pd.read_csv("../data/lego_sets.csv")

    eda()


if __name__ == "__main__":
    main()


# Use aggregate data to create market index
# Day by day of aggregate market stuff
# Less data with trees
# Group it by theme and use as market sector
# Don't have to trade, could always analyze something else with book value or something like that
