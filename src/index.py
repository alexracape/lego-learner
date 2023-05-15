# Script for constructing a Lego index / benchmark

import pandas as pd
import matplotlib.pyplot as plt


def get_market_weight_index(year, lag=2):
    """Gets a market weighted index of Lego sets for a given year"""

    # Read in data
    data = pd.read_csv("../data/custom_8.csv")
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop missing prices so can be evaluated
    data = data[data["Year"].between(year-lag, year)]  # Filter by year, do we want to include some previous years where prices are likely the same?

    # Get total market cap and set market caps
    data.loc[:, "Market_Cap"] = data["Current_Price"] * data["Owned"]
    total_market_cap = data["Market_Cap"].sum()
    data.loc[:, "Weight"] = data["Market_Cap"] / total_market_cap
    return data[["Set_ID", "Year", "Market_Cap", "Weight", "USD_MSRP", "Current_Price"]]


def get_equal_weighted_index(year, lag=2):
    """Gets an equal weighted index of Lego sets for a given year"""

    # Read in data
    data = pd.read_csv("../data/custom_8.csv")
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop missing prices so can be evaluated
    data = data[data["Year"].between(year-lag, year)]  # Filter by year, do we want to include some previous years where prices are likely the same?

    # Get total market cap and set market caps
    data["Market_Cap"] = data["Current_Price"].mul(data["Owned"])
    data["Weight"] = 1 / len(data)
    return data[["Set_ID", "Year", "Market_Cap", "Weight", "USD_MSRP", "Current_Price"]]


def get_index_return(index):
    """Helper to get the return of an index"""

    index["Return"] = index["Current_Price"] / index["USD_MSRP"] - 1
    index["Return_Weighted"] = index["Return"] * index["Weight"]
    return index["Return_Weighted"].sum()


def main():

    df = pd.DataFrame(columns=["Year", "Market_Weighted_Return", "Equal_Weighted_Return"])
    for year in range(2000, 2023):
        mw_index = get_market_weight_index(year)
        ew_index = get_equal_weighted_index(year)
        mw_return = get_index_return(mw_index)
        ew_return = get_index_return(ew_index)
        df = pd.concat([df, pd.DataFrame([[year, mw_return, ew_return]], columns=["Year", "Market_Weighted_Return", "Equal_Weighted_Return"])])

    df["Market_Weighted_Return"] = df["Market_Weighted_Return"] * 100
    df["Equal_Weighted_Return"] = df["Equal_Weighted_Return"] * 100
    df.plot(x="Year", y=["Market_Weighted_Return", "Equal_Weighted_Return"], ylabel="Return (%)", title="Market Weighted vs. Equal Weighted Index Returns")
    plt.show()


if __name__ == "__main__":
    main()
