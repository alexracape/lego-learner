
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BootstrapLearner import BootstrapLearner
from PERTLearner import PERTLearner
from index import get_market_weight_index, get_equal_weighted_index, get_index_return


def run_value_experiments():

    # Load data
    data = pd.read_csv("../data/custom_8.csv")
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    data = data.dropna(subset=["USD_MSRP"])  # Drop rows with missing prices
    data = data[["Year", "Pieces", "Theme", "Minifigures", "Rating", "Owned",
                 "USD_MSRP", "Current_Price"]]  # Note: took out current price to predict MSRP

    # Run an experiment for each year
    experiment_storage = pd.DataFrame(columns=["Year", "MWI", "EWI", "Portfolio"])
    for year in range(2000, 2024):
        # Train learner on all years before this one for value prediction
        training_data = data[data["Year"] < year]
        x_train = training_data.values[:, :-2]
        y_train = training_data.values[:, -2]
        learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=20)
        learner.train(x_train, y_train)

        # Test learner on this year's sets
        test_data = data[(data["Year"] == year) & (data["Current_Price"].notna())]
        x_test = test_data.values[:, :-2]
        predictions = learner.test(x_test)

        # Pick out portfolio with most undervalued sets
        value_differential = predictions - test_data["USD_MSRP"]  # Positive means undervalued
        portfolio = test_data[value_differential > 0]
        portfolio["Differential"] = value_differential[value_differential > 0]
        portfolio["Weight"] = portfolio["Differential"] / portfolio["Differential"].sum()  # Weight by differential
        # Evenly weight across top 10 biggest differntials
        # top_10 = portfolio.sort_values(by="Differential", ascending=False).head(10)
        # portfolio["Weight"] = 0
        # portfolio.loc[top_10.index, "Weight"] = 1 / 10

        # Evaluate how return stacks up against index
        mw_ind = get_market_weight_index(year, lag=0)
        ew_ind = get_equal_weighted_index(year, lag=0)
        mw_return = get_index_return(mw_ind)
        ew_return = get_index_return(ew_ind)
        portfolio_return = get_index_return(portfolio)  # need weight, current price, and list price
        results_df = pd.DataFrame([[year, mw_return, ew_return, portfolio_return]],
                                  columns=["Year", "MWI", "EWI", "Portfolio"])
        experiment_storage = pd.concat([experiment_storage, results_df])
        print(f"Finished experiment for {year}.")

    # Plot results
    plt.plot(experiment_storage["Year"], experiment_storage["MWI"], label="Market Weighted Index")
    plt.plot(experiment_storage["Year"], experiment_storage["EWI"], label="Equal Weighted Index")
    plt.plot(experiment_storage["Year"], experiment_storage["Portfolio"], label="Portfolio")
    plt.legend()
    plt.show()


def run_forecast_experiments():

    # Load data
    data = pd.read_csv("../data/custom_8.csv")
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop rows with missing prices
    data = data[["Year", "Pieces", "Theme", "Minifigures", "Rating", "Owned",
                 "USD_MSRP", "Current_Price"]]  # Note: took out current price to predict MSRP

    # Run an experiment for each year
    experiment_storage = pd.DataFrame(columns=["Year", "MWI", "EWI", "Portfolio"])
    for year in range(2000, 2024):
        # Train learner on all years before this one for value prediction
        training_data = data[data["Year"] < year]
        x_train = training_data.values[:, :-1]
        y_train = training_data.values[:, -1]
        learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=20)
        learner.train(x_train, y_train)

        # Test learner on this year's sets
        test_data = data[data["Year"] == year]
        x_test = test_data.values[:, :-1]
        predictions = learner.test(x_test)

        # Pick out portfolio with best predicted value
        portfolio = test_data.copy()
        portfolio["Prediction"] = predictions
        portfolio["Predicted_Gain"] = portfolio["Prediction"] / portfolio["USD_MSRP"] - 1
        portfolio["Weight"] = portfolio["Predicted_Gain"] / portfolio["Predicted_Gain"].sum()  # Weight by gain
        # Evenly weight across top 10 biggest differntials
        # top_10 = portfolio.sort_values(by="Differential", ascending=False).head(10)
        # portfolio["Weight"] = 0
        # portfolio.loc[top_10.index, "Weight"] = 1 / 10

        # Evaluate how return stacks up against index
        mw_ind = get_market_weight_index(year, lag=0)
        ew_ind = get_equal_weighted_index(year, lag=0)
        mw_return = get_index_return(mw_ind)
        ew_return = get_index_return(ew_ind)
        portfolio_return = get_index_return(portfolio)  # need weight, current price, and list price
        results_df = pd.DataFrame([[year, mw_return, ew_return, portfolio_return]],
                                  columns=["Year", "MWI", "EWI", "Portfolio"])
        experiment_storage = pd.concat([experiment_storage, results_df])
        print(f"Finished experiment for {year}.")

    # Plot results
    plt.plot(experiment_storage["Year"], experiment_storage["MWI"], label="Market Weighted Index")
    plt.plot(experiment_storage["Year"], experiment_storage["EWI"], label="Equal Weighted Index")
    plt.plot(experiment_storage["Year"], experiment_storage["Portfolio"], label="Portfolio")
    plt.legend()
    plt.show()


def main():
    #run_value_experiments()
    run_forecast_experiments()


if __name__ == "__main__":
    main()

