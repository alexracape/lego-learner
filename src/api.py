# Methods and scripts to extract data from api's and merge with csv
# Outline of the process so far:
# 1. Started with old_data_source.csv and used the ID's to get the current price from Bricklist API
# 2. Tried kaggle data set for list price -> only had 800 sets, many duplicates because scraped each international site
# 3. Used Brickset API to get the list price for each set -> Most sets returned no price data (only 800 returned prices)
# 4. Found a github repo after deep dive into reddit with a csv built for R
# 5. Used that as a new base and merged in the current price from previous Bricklink API Scrape
# 6. Realized this dataset only went up to 2015 -> Use brickset API to get updated sets data 2016-2023
# 7. Realized that original data source only went to 2018 -> rescrape some more Bricklink data to get 2019-2023
# 8. Realized that API was limiting us to 50 matches, and we needed number owned to construct index
# 9. Used both API's to scrape dataset from scratch

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
    api_endpoint = f"https://api.bricklink.com/api/store/v1/items/SET/{set_id}/price"
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
        quantity = response["data"]["total_quantity"] if "total_quantity" in response["data"] else np.nan
    except Exception as e:
        quantity = np.nan

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
        print(f"{set_id} Found Set")
        return most_recent_order["unit_price"], quantity
    except Exception as e:
        print(f"Uh oh exception: {e} for {set_id} (probably resource couldn't be found)")
        return np.nan, quantity


def get_all_current_prices(df):
    """Get all current prices for all sets in the database"""

    df["Total_Quantity"] = np.nan
    df["Current_Price"] = np.nan
    for i in range(len(df)):
        set_id = df["Set_ID"][i]
        current_price, total_quantity = get_current_price(set_id)
        df.loc[i, "Current_Price"] = current_price
        df.loc[i, "Total_Quantity"] = total_quantity

    df.to_csv("custom_8.csv", index=False)


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


def get_data_by_year(year, df, user_hash):

    # Set the API endpoint and parameters
    api_endpoint = "https://brickset.com/api/v3.asmx/getSets"
    params = {
        "apiKey": BS_API_KEY,
        "userHash": user_hash,
        "params": json.dumps({
            "year": year,
            "pageSize": 500,
        })
    }

    # Send the API request
    response = requests.get(api_endpoint, params=params)
    response = response.json()
    # response = {'status': 'success', 'matches': 45, 'sets': [{'setID': 5261, 'number': '75', 'numberVariant': 1, 'name': 'PreSchool Set', 'year': 1975, 'theme': 'PreSchool', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 16, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/75-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/75-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/75-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 10, 'wantedBy': 28}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-05T13:20:02.207Z'}, {'setID': 5262, 'number': '77', 'numberVariant': 1, 'name': 'PreSchool Set', 'year': 1975, 'theme': 'PreSchool', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 20, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/77-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/77-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/77-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 11, 'wantedBy': 27}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2012-02-15T05:09:21.747Z'}, {'setID': 30975, 'number': '077', 'numberVariant': 1, 'name': 'Pre-School Set', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 21, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/077-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/077-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/077-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 10, 'wantedBy': 6}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 1, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2020-11-17T08:17:05.9Z'}, {'setID': 5263, 'number': '78', 'numberVariant': 1, 'name': 'PreSchool Set', 'year': 1975, 'theme': 'PreSchool', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 32, 'image': {}, 'bricksetURL': 'https://brickset.com/sets/78-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 8, 'wantedBy': 25}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2005-12-27T02:22:05Z'}, {'setID': 22707, 'number': '78', 'numberVariant': 3, 'name': 'Basic Set', 'year': 1975, 'theme': 'Samsonite', 'themeGroup': 'Vintage', 'subtheme': 'Basic set', 'category': 'Normal', 'released': True, 'pieces': 330, 'image': {}, 'bricksetURL': 'https://brickset.com/sets/78-3', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 10, 'wantedBy': 28}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2013-04-11T00:00:00Z'}, {'setID': 7985, 'number': '133', 'numberVariant': 1, 'name': 'Locomotive', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 87, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/133-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/133-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/133-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 279, 'wantedBy': 180}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-05T16:23:50.93Z'}, {'setID': 252, 'number': '134', 'numberVariant': 1, 'name': 'Mobile Crane and Wagon', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 86, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/134-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/134-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/134-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 252, 'wantedBy': 179}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-06T11:48:49.187Z'}, {'setID': 261, 'number': '136', 'numberVariant': 1, 'name': 'Tanker Wagon', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 83, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/136-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/136-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/136-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 404, 'wantedBy': 199}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-06T11:50:24.907Z'}, {'setID': 7437, 'number': '137', 'numberVariant': 2, 'name': 'Passenger Sleeping Car', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 82, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/137-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/137-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/137-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 299, 'wantedBy': 177}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-06T11:53:47.633Z'}, {'setID': 352, 'number': '148', 'numberVariant': 1, 'name': 'Station', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 293, 'minifigs': 5, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/148-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/148-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/148-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 479, 'wantedBy': 300}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 1, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-06T12:04:57.743Z'}, {'setID': 608, 'number': '182', 'numberVariant': 1, 'name': 'Train Set with Motor', 'year': 1975, 'theme': 'Trains', 'themeGroup': 'Modern day', 'subtheme': '4.5/12V', 'category': 'Normal', 'released': True, 'pieces': 369, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/182-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/182-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/182-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 409, 'wantedBy': 211}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 1, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-06T14:10:46.62Z'}, {'setID': 5244, 'number': '190', 'numberVariant': 1, 'name': 'Farm Set', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 526, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/190-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/190-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/190-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 223, 'wantedBy': 127}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 1, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2005-12-15T23:56:27Z'}, {'setID': 5269, 'number': '195', 'numberVariant': 1, 'name': 'Airplane', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 89, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/195-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/195-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/195-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 145, 'wantedBy': 82}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2011-08-31T14:24:03.84Z'}, {'setID': 828, 'number': '222', 'numberVariant': 1, 'name': 'Building Ideas Book', 'year': 1975, 'theme': 'Books', 'themeGroup': 'Miscellaneous', 'subtheme': 'LEGO', 'category': 'Book', 'released': True, 'pieces': 1, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/222-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/222-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/222-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 695, 'wantedBy': 122}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2020-11-10T14:14:47.43Z'}, {'setID': 931, 'number': '253', 'numberVariant': 2, 'name': 'Helicopter and Pilot', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 49, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/253-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/253-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/253-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 204, 'wantedBy': 76}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 2, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-07T12:13:51.407Z'}, {'setID': 7438, 'number': '254', 'numberVariant': 1, 'name': 'Family', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 103, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/254-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/254-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/254-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 264, 'wantedBy': 80}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-07T12:18:28.19Z'}, {'setID': 7439, 'number': '255', 'numberVariant': 2, 'name': 'Farming Scene', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 120, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/255-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/255-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/255-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 134, 'wantedBy': 84}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-07T12:23:04.46Z'}, {'setID': 6539, 'number': '362', 'numberVariant': 1, 'name': 'Windmill', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 215, 'minifigs': 2, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/362-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/362-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/362-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 280, 'wantedBy': 189}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 2, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T12:00:57.94Z'}, {'setID': 6368, 'number': '363', 'numberVariant': 1, 'name': 'Hospital', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 229, 'minifigs': 7, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/363-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/363-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/363-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 641, 'wantedBy': 237}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 4.2, 'reviewCount': 3, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T12:04:04.65Z'}, {'setID': 7440, 'number': '364', 'numberVariant': 1, 'name': 'Harbour', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 518, 'minifigs': 7, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/364-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/364-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/364-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 188, 'wantedBy': 215}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T12:05:48.943Z'}, {'setID': 7441, 'number': '365', 'numberVariant': 1, 'name': 'Wild West', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 602, 'minifigs': 8, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/365-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/365-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/365-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 255, 'wantedBy': 226}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 1, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T15:27:54.237Z'}, {'setID': 6023, 'number': '367', 'numberVariant': 1, 'name': 'Space Module with Astronauts', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 364, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/367-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/367-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/367-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 597, 'wantedBy': 331}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 3.7, 'reviewCount': 3, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T15:30:00.12Z'}, {'setID': 6024, 'number': '390', 'numberVariant': 2, 'name': '1913 Cadillac', 'year': 1975, 'theme': 'Hobby Set', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 200, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/390-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/390-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/390-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 762, 'wantedBy': 454}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 3.9, 'reviewCount': 2, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T16:09:29.21Z'}, {'setID': 6025, 'number': '391', 'numberVariant': 1, 'name': '1926 Renault', 'year': 1975, 'theme': 'Hobby Set', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 237, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/391-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/391-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/391-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 609, 'wantedBy': 461}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 4.3, 'reviewCount': 2, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T16:10:48.28Z'}, {'setID': 6026, 'number': '392', 'numberVariant': 1, 'name': 'Formula 1', 'year': 1975, 'theme': 'Hobby Set', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 197, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/392-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/392-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/392-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 443, 'wantedBy': 378}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 2, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-10T16:12:36.88Z'}, {'setID': 5275, 'number': '430', 'numberVariant': 1, 'name': 'Biplane', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 18, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/430-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/430-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/430-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 454, 'wantedBy': 116}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 4.0, 'reviewCount': 5, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2012-07-04T11:02:13.287Z'}, {'setID': 5276, 'number': '480', 'numberVariant': 1, 'name': 'Rescue Helicopter', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 62, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/480-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/480-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/480-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 410, 'wantedBy': 120}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:28:11.397Z'}, {'setID': 5278, 'number': '490', 'numberVariant': 1, 'name': 'Mobile Crane', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 46, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/490-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/490-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/490-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 383, 'wantedBy': 110}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:30:56.48Z'}, {'setID': 7443, 'number': '516', 'numberVariant': 1, 'name': 'Bricks and half bricks', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 20, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/516-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/516-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/516-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 17, 'wantedBy': 24}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Promotional', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:39:53.813Z'}, {'setID': 7444, 'number': '517', 'numberVariant': 1, 'name': 'Bricks and half bricks and arches', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 121, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/517-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/517-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/517-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 29, 'wantedBy': 25}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2010-04-05T04:11:20.013Z'}, {'setID': 7445, 'number': '518', 'numberVariant': 8, 'name': 'Bricks and half bricks and trolley', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 17, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/518-8.jpg', 'imageURL': 'https://images.brickset.com/sets/images/518-8.jpg'}, 'bricksetURL': 'https://brickset.com/sets/518-8', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 24, 'wantedBy': 24}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:45:11.34Z'}, {'setID': 7446, 'number': '519', 'numberVariant': 8, 'name': 'Bricks and half bricks and arches and trolley', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 240, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/519-8.jpg', 'imageURL': 'https://images.brickset.com/sets/images/519-8.jpg'}, 'bricksetURL': 'https://brickset.com/sets/519-8', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 21, 'wantedBy': 27}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:49:08.52Z'}, {'setID': 7447, 'number': '520', 'numberVariant': 9, 'name': 'Bricks and half bricks and two tolleys', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 168, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/520-9.jpg', 'imageURL': 'https://images.brickset.com/sets/images/520-9.jpg'}, 'bricksetURL': 'https://brickset.com/sets/520-9', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 24, 'wantedBy': 27}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:51:38.1Z'}, {'setID': 7449, 'number': '521', 'numberVariant': 8, 'name': 'Bricks and half bricks all colours', 'year': 1975, 'theme': 'Duplo', 'themeGroup': 'Pre-school', 'category': 'Normal', 'released': True, 'pieces': 640, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/521-8.jpg', 'imageURL': 'https://images.brickset.com/sets/images/521-8.jpg'}, 'bricksetURL': 'https://brickset.com/sets/521-8', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 29, 'wantedBy': 25}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T10:53:03.463Z'}, {'setID': 5284, 'number': '580', 'numberVariant': 1, 'name': 'Brick Yard', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 216, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/580-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/580-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/580-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 368, 'wantedBy': 137}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 1, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2012-07-04T11:02:31.46Z'}, {'setID': 6174, 'number': '615', 'numberVariant': 2, 'name': 'Forklift', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'subtheme': 'Vehicle', 'category': 'Normal', 'released': True, 'pieces': 21, 'minifigs': 1, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/615-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/615-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/615-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 1019, 'wantedBy': 140}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 2, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T11:42:10.19Z'}, {'setID': 5098, 'number': '659', 'numberVariant': 1, 'name': 'Police Patrol', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 49, 'minifigs': 2, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/659-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/659-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/659-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 953, 'wantedBy': 154}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 3.7, 'reviewCount': 2, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T13:25:38.917Z'}, {'setID': 5107, 'number': '660', 'numberVariant': 1, 'name': 'Air Transporter', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 39, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/660-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/660-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/660-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 706, 'wantedBy': 137}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 1, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T13:28:24.653Z'}, {'setID': 5099, 'number': '692', 'numberVariant': 1, 'name': 'Road Repair Crew', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 57, 'minifigs': 2, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/692-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/692-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/692-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 486, 'wantedBy': 122}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2013-02-24T14:09:17.51Z'}, {'setID': 5095, 'number': '693', 'numberVariant': 1, 'name': 'Fire engine with firemen', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 62, 'minifigs': 3, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/693-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/693-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/693-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 533, 'wantedBy': 134}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {'min': 6}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2023-02-25T15:02:51.853Z'}, {'setID': 6700, 'number': '694', 'numberVariant': 1, 'name': 'Transport Truck', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'subtheme': 'Vehicle', 'category': 'Normal', 'released': True, 'pieces': 65, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/694-1.jpg', 'imageURL': 'https://images.brickset.com/sets/images/694-1.jpg'}, 'bricksetURL': 'https://brickset.com/sets/694-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 475, 'wantedBy': 148}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-12T13:51:30.433Z'}, {'setID': 5231, 'number': '760', 'numberVariant': 2, 'name': 'London Bus', 'year': 1975, 'theme': 'LEGOLAND', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 110, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/760-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/760-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/760-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 404, 'wantedBy': 233}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 2, 'packagingType': '{Not specified}', 'availability': '{Not specified}', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2017-01-09T08:28:54.11Z'}, {'setID': 7442, 'number': '814', 'numberVariant': 2, 'name': 'Tractor', 'year': 1975, 'theme': 'Building Set with People', 'themeGroup': 'Vintage', 'category': 'Normal', 'released': True, 'pieces': 93, 'image': {'thumbnailURL': 'https://images.brickset.com/sets/small/814-2.jpg', 'imageURL': 'https://images.brickset.com/sets/images/814-2.jpg'}, 'bricksetURL': 'https://brickset.com/sets/814-2', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 138, 'wantedBy': 101}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2021-05-13T11:40:14.107Z'}, {'setID': 9532, 'number': 'BB173', 'numberVariant': 1, 'name': 'Holdall Storage Bag', 'year': 1975, 'theme': 'Gear', 'themeGroup': 'Miscellaneous', 'subtheme': 'Storage', 'category': 'Gear', 'released': True, 'image': {}, 'bricksetURL': 'https://brickset.com/sets/BB173-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 11, 'wantedBy': 27}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Box', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2012-03-31T08:01:46.14Z'}, {'setID': 9361, 'number': 'SUPSET1', 'numberVariant': 1, 'name': 'Super Set', 'year': 1975, 'theme': 'Universal Building Set', 'themeGroup': 'Basic', 'category': 'Other', 'released': True, 'image': {}, 'bricksetURL': 'https://brickset.com/sets/SUPSET1-1', 'collection': {'owned': False, 'wanted': False, 'qtyOwned': 0, 'rating': 0, 'notes': ''}, 'collections': {'ownedBy': 15, 'wantedBy': 37}, 'LEGOCom': {'US': {}, 'UK': {}, 'CA': {}, 'DE': {}}, 'rating': 0.0, 'reviewCount': 0, 'packagingType': 'Other', 'availability': 'Retail', 'instructionsCount': 0, 'additionalImageCount': 0, 'ageRange': {}, 'dimensions': {}, 'barcode': {}, 'extendedData': {}, 'lastUpdated': '2012-02-18T08:16:31.683Z'}]}
    if "message" in response:
        print("API Limit Exceeded")
        return None

    # Lots more features in here if we need it
    try:
        sets = response["sets"]
        rows = []
        for set in sets:

            # Id / Number
            set_number = set["number"]
            set_var = set["numberVariant"]
            set_id = f"{set_number}-{set_var}"

            # Basic common info
            set_name = set["name"]
            set_year = set["year"]
            set_pieces = set["pieces"] if "pieces" in set else np.nan
            set_minifigs = set["minifigs"] if "minifigs" in set else np.nan
            set_packaging = set["packagingType"] if "packagingType" in set else np.nan
            set_rating = set["rating"] if "rating" in set else np.nan
            set_availability = set["availability"] if "availability" in set else np.nan
            set_num_instructions = set["instructionsCount"] if "instructionsCount" in set else np.nan

            # Theme
            set_theme = set["theme"] if "theme" in set else np.nan
            set_theme_group = set["themeGroup"] if "themeGroup" in set else np.nan
            set_subtheme = set["subtheme"] if "subtheme" in set else np.nan
            set_category = set["category"] if "category" in set else np.nan

            # Num Owned
            try:
                num_owned = set["collections"]["ownedBy"]
            except Exception as e:
                num_owned = np.nan

            # List price
            try:
                price = set["LEGOCom"]["US"]["retailPrice"]
            except Exception as e:
                price = np.nan

            row = {
                "Set_ID": set_id,
                "Name": set_name,
                "Year": set_year,
                "Theme": set_theme,
                "Theme_Group": set_theme_group,
                "Subtheme": set_subtheme,
                "Category": set_category,
                "Packaging": set_packaging,
                "Num_Instructions": set_num_instructions,
                "Availability": set_availability,
                "Pieces": set_pieces,
                "Minifigures": set_minifigs,
                "Owned": num_owned,
                "Rating": set_rating,
                "USD_MSRP": price,
            }
            rows.append(row)

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    except Exception as e:
        print(f"Could not find matches for {year} -- {e}")

    # Indicate that API limit has not yet been reached
    return df


def from_scratch():
    """Get num_owned for every set possible from Brickset API"""

    df = pd.DataFrame(columns=["Set_ID", "Name", "Year", "Theme", "Theme_Group", "Subtheme", "Category", "Packaging",
                               "Num_Instructions", "Availability", "Pieces", "Minifigures", "Owned", "Rating",
                               "USD_MSRP"])
    user_hash = get_user_hash(BS_USERNAME, BS_PASSWORD)
    for year in range(1975, 2024):
        df = get_data_by_year(year, df, user_hash)
        if df is None:
            print(f"API LIMIT REACHED: stopped at {year}")
            break
        else:
            print(f"Finished {year}")

    df.to_csv("custom_7.csv", index=False)


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


def eda(data_source="custom_8.csv"):
    """Some basic EDA to explore the scraped sets / price data"""
    line, = plt.plot([], [], color='black', linewidth=2, label='S&P 500')
    # Create the legend
    plt.legend(handles=[line])

    # Load data
    data = pd.read_csv(f"../data/{data_source}")

    # Check how much data has both list and current price
    with_both = data[(data["Current_Price"].notna()) & (data["USD_MSRP"].notna())]
    with_current = data[data["Current_Price"].notna()]
    print(f"Data with both list and current price: {len(with_both)}")

    # Plot the price vs year and histogram of price
    ax = data.plot(x="Year", y="Current_Price", kind="scatter", logy=True, xlabel="Year", ylabel="Current Price")
    ax.annotate("Death Star", xy=(2005, 1800), xytext=(2010, 5000), color='black', arrowprops={'facecolor':'black', 'edgecolor':'none'})
    ax.annotate("Brick Seperator", xy=(2011, 0.05), xytext=(1995, .1), color='black', arrowprops={'facecolor':'black', 'edgecolor':'none'})
    ax.annotate("Crusader's Cart", xy=(1990, 395.00), xytext=(1982, 4000.00), color='black', arrowprops={'facecolor':'black', 'edgecolor':'none'})
    ax.annotate("Shell Tanker", xy=(1980, 189.00), xytext=(1975, 1000.00), color='black', arrowprops={'facecolor':'black', 'edgecolor':'none'})
    ax.annotate("Luke's Speeder", xy=(2022, 149.00), xytext=(2016, 2000.00), color='black', arrowprops={'facecolor':'black', 'edgecolor':'none'})

    # data.plot(y="Current_Price", kind="hist", logy=True, bins=1000)
    # fig, ax = plt.subplots()
    # ax.hist2d(with_current["Year"], with_current["Current_Price"], bins=100)
    plt.show()


def main():
    """Main function for simple tests"""

    # Load Data Sets
    base = pd.read_csv("../data/custom_8.csv")
    with_list = base[base["USD_MSRP"].notna()]  # 5,837
    with_current = base[base["Current_Price"].notna()]  # 5,442
    with_both = base[(base["Current_Price"].notna()) & (base["USD_MSRP"].notna())]  # 3,612

    # Get data from brickset: ie custom 7+ base
    # from_scratch()
    #get_all_current_prices(base)

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
