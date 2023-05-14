# Script for constructing a Lego index / benchmark

import pandas as pd
import matplotlib.pyplot as plt


def get_market_weight_index(year):
    """Gets a market weighted index of Lego sets for a given year"""

    # Read in data
    data = pd.read_csv("../data/custom_8.csv")
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop missing prices so can be evaluated
    data = data[data["Year"].between(year, year)]  # Filter by year, do we want to include some previous years where prices are likely the same?

    # Get total market cap and set market caps
    data["Market_Cap"] = data["Current_Price"] * data["Owned"]
    total_market_cap = data["Market_Cap"].sum()
    data["Weight"] = data["Market_Cap"] / total_market_cap
    return data[["Set_ID", "Year", "Market_Cap", "Weight", "USD_MSRP", "Current_Price"]]


def get_equal_weighted_index(year):
    """Gets an equal weighted index of Lego sets for a given year"""

    # Read in data
    data = pd.read_csv("../data/custom_8.csv")
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop missing prices so can be evaluated
    data = data[data["Year"].between(year, year)]  # Filter by year, do we want to include some previous years where prices are likely the same?

    # Get total market cap and set market caps
    data["Market_Cap"] = data["Current_Price"] * data["Owned"]
    data["Weight"] = 1 / len(data)
    return data[["Set_ID", "Year", "Market_Cap", "Weight", "USD_MSRP", "Current_Price"]]


def get_index_return(index):
    """Helper to get the return of an index"""

    index["Return"] = index["Current_Price"] / index["USD_MSRP"] - 1
    index["Return_Weighted"] = index["Return"] * index["Weight"]
    return index["Return_Weighted"].sum()


def main():

    for year in range(2000, 2023):
        mw_index = get_market_weight_index(year)
        ew_index = get_equal_weighted_index(year)
        mw_return = get_index_return(mw_index)
        ew_return = get_index_return(ew_index)
        print(f"{year} Market weighted index return: {mw_return*100}%")
        print(f"{year} Equal weighted index return: {ew_return*100}%")


if __name__ == "__main__":
    main()
