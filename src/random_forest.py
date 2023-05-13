
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BootstrapLearner import BootstrapLearner
from PERTLearner import PERTLearner


def main():

    # Set up data
    data = pd.read_csv("../data/custom_6.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    # counts = data["Theme"].value_counts().to_dict()  # Frequency encoding approach, some themes have same count tho
    # data["Theme"] = data["Theme"].map(counts)
    data = data.dropna()  # Drop rows with missing data - 1939 remaining...
    clean = data[["Year", "Pieces", "Minifigures", "Theme", "USD_MSRP", "Current_Price"]]
    data = clean.to_numpy()

    # Run experiments to tune hyperparameters
    bag_vals = range(1, 50, 5)
    experiment_storage = np.zeros((len(bag_vals), 5))
    for num_bags in bag_vals:
        print("Running experiment with {} bags".format(num_bags))

        # Set up learner
        learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=20)

        # Set up cross validation
        num_folds = 5
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

        # Print Results
        print(f"Mean Squared Error (in sample): {mean_rmse_is}\n\t{rmse_in_sample_scores}")
        print(f"Correlation (in sample): {mean_correlation_is}\n\t{correlation_in_sample_scores}")
        print(f"Mean Squared Error: {mean_rmse_os}\n\t{rmse_scores}")
        print(f"Correlation: {mean_correlation_os}\n\t{correlation_scores}")
        # First try: is_corr = .96, os_corr = .65 so way better than neural net for now

        # Store results
        experiment_storage[num_bags // 5 - 1, 0] = num_bags
        experiment_storage[num_bags // 5 - 1, 1] = mean_rmse_is
        experiment_storage[num_bags // 5 - 1, 2] = mean_correlation_is
        experiment_storage[num_bags // 5 - 1, 3] = mean_rmse_os
        experiment_storage[num_bags // 5 - 1, 4] = mean_correlation_os

    # Plot experiment results
    exp_data = pd.DataFrame(experiment_storage, columns=["Bags", "IS RMSE", "IS Correlation", "OS RMSE", "OS Correlation"])
    exp_data.plot(x="Bags", y=["IS RMSE", "OS RMSE"])
    plt.show()


if __name__ == "__main__":
    main()
