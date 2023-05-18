
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BootstrapLearner import BootstrapLearner
from PERTLearner import PERTLearner


def run_experiment(learner, data, num_folds=3):
    """Method to train and test learner with certain hypers

    Uses k-fold cross validation
    """

    # Set up cross validation
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Train and evaluate for each fold
    rmse_scores = []
    correlation_scores = []
    rmse_in_sample_scores = []
    correlation_in_sample_scores = []
    for train_indices, test_indices in kf.split(data):
        # Get split
        x_train, y_train = data[train_indices, :-1], data[train_indices, -1]
        x_test, y_test = data[test_indices, :-1], data[test_indices, -1]

        # Train and fit
        learner.train(x_train, y_train)

        # Test in sample
        in_sample_predictions = learner.test(x_train)
        rmse_in_sample = mean_squared_error(y_train, in_sample_predictions, squared=False)
        correlation_in_sample = np.corrcoef(y_train.astype(np.float64), in_sample_predictions)[0, 1]
        rmse_in_sample_scores.append(rmse_in_sample)
        correlation_in_sample_scores.append(correlation_in_sample)

        # Test out of sample
        predictions = learner.test(x_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        correlation = np.corrcoef(y_test.astype(np.float64), predictions)[0, 1]
        rmse_scores.append(rmse)
        correlation_scores.append(correlation)

    # Compute the mean scores over all folds
    mean_rmse_os = np.mean(rmse_scores)
    mean_correlation_os = np.mean(correlation_scores)
    mean_rmse_is = np.mean(rmse_in_sample_scores)
    mean_correlation_is = np.mean(correlation_in_sample_scores)

    # Return results
    return mean_rmse_is, mean_correlation_is, mean_rmse_os, mean_correlation_os


def main():

    # Set up data
    data = pd.read_csv("../data/custom_8.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    # counts = data["Theme"].value_counts().to_dict()  # Frequency encoding approach, some themes have same count tho
    # data["Theme"] = data["Theme"].map(counts)
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop rows with missing data
    clean = data[["Year", "Theme", "Pieces", "Minifigures", "Rating", "Owned", "USD_MSRP", "Current_Price"]]
    data = clean.to_numpy()

    # Run experiments to tune hyperparameters
    bag_vals = list(range(1, 10)) + list(range(10, 50, 5))
    experiment_storage = np.zeros((len(bag_vals), 5))
    for num_bags, i in zip(bag_vals, range(len(bag_vals))):
        print(f"Running experiment with {num_bags} bags")

        # Set up learner
        learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=num_bags)

        # Run experiment
        results = run_experiment(learner, data)
        experiment_storage[i, 0] = num_bags
        experiment_storage[i, 1:5] = results

    # Plot experiment results
    exp_data = pd.DataFrame(experiment_storage, columns=["Bags", "IS RMSE", "IS Correlation", "OS RMSE", "OS Correlation"])
    exp_data.plot(x="Bags", y=["IS RMSE", "OS RMSE"])
    plt.xlabel("Number of Bags")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Number of Bags for Random Forest Price Prediction")
    plt.show()

    # # Get best model and run some tests on it to explore whats going on
    # bag_best_hyper = int(experiment_storage[np.argmin(experiment_storage[:, 3]), 0])
    bag_best_hyper = 20
    learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=bag_best_hyper)
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)
    learner.train(x_train, y_train)
    predictions = learner.test(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    correlation = np.corrcoef(y_test.astype(np.float64), predictions)[0, 1]
    r_squared = r2_score(y_test, predictions)

    # Print some summary stats
    print(f"Best model has {bag_best_hyper} bags")
    print(f"RMSE: {rmse}")
    print(f"Correlation: {correlation}")
    print(f"R^2: {r_squared}")

    # Only a couple of very large outliers
    residuals = y_test - predictions
    plt.figure()
    plt.scatter(predictions, residuals)
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residuals for Random Forest MSRP Prediction")
    plt.show()

    # Pick out biggest outliers
    outliers = np.argsort(residuals)[-10:]
    outlier_features = x_test[outliers, :]
    print(f"Biggest outliers are {outlier_features}")


if __name__ == "__main__":
    main()
