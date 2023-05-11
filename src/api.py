# Methods and scripts to extract data from api's and merge with csv
# Outline of the process so far:
# 1. Started with old_data_source.csv and used the ID's to get the current price from Bricklist API
# 2. Tried kaggle data set for list price -> only had 800 sets, many duplicates because scraped each international site
# 3. Used Brickset API to get the list price for each set -> Most sets returned no price data (only 800 returned prices)
# 4. Found a github repo after deep dive into reddit with a csv built for R
# 5. Used that as a new base and merged in the current price from previous Bricklink API Scrape
# 6. Realized this dataset only went up to 2015 -> Use brickset API to get updated sets data 2016-2023
# 7. Realized that original data source only went to 2018 -> rescrape some more Bricklink data to get 2019-2023

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
        return None

    # Lots more features in here if we need it
    try:
        sets = response["sets"]
        for set in sets:
            set_number = set["number"]
            set_name = set["name"]
            set_year = set["year"]
            set_theme = set["theme"] if "theme" in set else np.nan
            set_subtheme = set["subtheme"] if "subtheme" in set else np.nan
            set_pieces = set["pieces"] if "pieces" in set else np.nan
            set_minifigs = set["minifigs"] if "minifigs" in set else np.nan
            try:
                price = set["LEGOCom"]["US"]["retailPrice"]
            except Exception as e:
                print(f"Could not find price for {set_number} -- {e}")
                price = np.nan
            row = {
                "Item_Number": set_number,
                "Name": set_name,
                "Year": set_year,
                "Theme": set_theme,
                "Subtheme": set_subtheme,
                "Pieces": set_pieces,
                "Minifigures": set_minifigs,
                "USD_MSRP": price,
            }
            df = df.append(row, ignore_index=True)

    except Exception as e:
        print(f"Could not find matches for {year} -- {e}")

    # Indicate that API limit has not yet been reached
    return df


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
    api_endpoint = f"https://api.bricklink.com/api/store/v1/items/SET/{set_id}-1/price"
    oauth = OAuth1(
        client_key=CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=BL_API_TOKEN,
        resource_owner_secret=BL_API_SECRET,
    )
    params = {
        "guide_type": "sold",  # sold (closed) or stock (active listings)
        "no": set_id,
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


def get_all_list_prices(df):
    """Use Brickset to get list prices from lego.com

    Could not be solely used since the API was returning 0's for almost all sets before 2015 ish
    - got 788 prices
    - only 175 overlapped with current trades
    However it was used to fill in the gaps from the other data source
    """
    user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    for year in range(2016, 2024):
        df = get_prices_by_year(year, df, user_hash)
        if df is None:
            print(f"API LIMIT REACHED: stopped at {year}")
            return df
        else:
            print(f"Finished {year}")
    return df


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


def eda(data_source="custom_6.csv"):
    """Some basic EDA to explore the scraped sets / price data"""

    # Load data
    data = pd.read_csv(f"../data/{data_source}")

    # Check how much data has both list and current price
    with_both = data[(data["Current_Price"].notna()) & (data["USD_MSRP"].notna())]
    print(f"Data with both list and current price: {len(with_both)}")

    # Plot the price vs year and histogram of price
    data.plot(x="Year", y="Current_Price", kind="scatter", logy=True, xlabel="Year", ylabel="Current Price")
    data.plot(y="Current_Price", kind="hist", logy=True, bins=100)
    data.plot(x="Year", y="USD_MSRP", kind="scatter", logy=True, xlabel="Year", ylabel="List Price")
    plt.show()


def main():
    """Main function for simple tests"""

    # Load Data Sets
    base = pd.read_csv("../data/custom_6.csv")
    old_custom = pd.read_csv("../data/custom_5.csv")
    lego_sets = pd.read_csv("../data/lego_sets.csv")
    # base = base.drop_duplicates(subset=["Item_Number"], keep="first")

    # Rescrape recent current price data
    # ids = base["Item_Number"].tolist()
    # base = base.set_index("Item_Number")
    # for set_id in ids:
    #     current_price = get_current_price(set_id)
    #     base.loc[set_id, "Current_Price"] = current_price
    #
    # base["Current_Price"] = base["Current_Price"].replace(0, np.nan)
    # base.to_csv("custom_6.csv", index=True)

    # Idea: after run for 6 is complete, can merge in new current prices with the old ones and get best of both
    # can also check that they are equal /

    # Combine in recent data from 2015 onwards
    # base.drop("current_price", axis=1, inplace=True)
    # base = get_all_list_prices(base)

    # Clean up and merge dataframes
    # old_custom[['set_num', 'variation', 'left_over']] = old_custom['set_num'].str.split('-', expand=True)
    # current_price_df = old_custom[["set_num", "current_price"]]
    # base = pd.merge(base, current_price_df, left_on="Item_Number", right_on="set_num", how="left")
    # base.drop('set_num', axis=1, inplace=True)
    # base["current_price"] = base["current_price"].replace(0, np.nan)
    # base.rename(columns={"current_price": "Current_Price"}, inplace=True)
    # base.to_csv("custom_5.csv", index=False)

    eda()


if __name__ == "__main__":
    main()
