# Script to explore the correlations Between features and output
# Idea: use fraction of explained variance to see contribution of each feature using permutation
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np

from BootstrapLearner import BootstrapLearner
from PERTLearner import PERTLearner


def random_forest_feature_importance():

    # Set up data
    data = pd.read_csv("../data/custom_8.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop rows with missing data
    features = ["Year", "Pieces", "Theme", "Minifigures", "Rating", "Owned", "USD_MSRP"]
    clean = data[features]
    data = clean.to_numpy()

    # Rotate through features to test
    num_features = data.shape[1] - 1
    num_rows = data.shape[0]
    experiment_results = pd.DataFrame(columns=["Feature", "R2_IS", "R2_OS"])
    for i in range(num_features + 1):

        # If we are testing a feature, scramble it, else test normal data
        feature_data = data.copy()
        if i < num_features:
            feature_data[:, i] = feature_data[:, i][np.random.permutation(num_rows)]

        # Set up learner
        learner = BootstrapLearner(constituent=PERTLearner, kwargs={}, bags=20)

        # Set up cross validation
        kf = KFold(n_splits=5, shuffle=True)

        # Train and evaluate for each fold
        r2_is_scores = []
        r2_os_scores = []
        for train_indices, test_indices in kf.split(feature_data):
            # Get split
            x_train, y_train = feature_data[train_indices, :-1], feature_data[train_indices, -1]
            x_test, y_test = feature_data[test_indices, :-1], feature_data[test_indices, -1]

            # Train and fit
            learner.train(x_train, y_train)

            # Test in sample
            in_sample_predictions = learner.test(x_train)
            r_squared_in_sample = r2_score(y_train, in_sample_predictions)
            r2_is_scores.append(r_squared_in_sample)

            # Test out of sample
            predictions = learner.test(x_test)
            r_squared = r2_score(y_test, predictions)
            r2_os_scores.append(r_squared)

        # Compute the mean R^2 score
        mean_r2_is = np.mean(r2_is_scores)
        mean_r2_os = np.mean(r2_os_scores)
        feature_results = pd.DataFrame({"Feature": features[i], "R2_IS": mean_r2_is, "R2_OS": mean_r2_os}, index=[0])
        experiment_results = pd.concat([experiment_results, feature_results], ignore_index=True)
        print(f"Finished testing feature: '{features[i]}'")

    # Print results
    print(experiment_results)
    importances = experiment_results
    importances["In Sample"] = importances["R2_IS"][6] - importances["R2_IS"]
    importances["Out of Sample"] = importances["R2_OS"][6] - importances["R2_OS"]
    importances = importances[importances["Feature"] != "USD_MSRP"]
    importances.plot.bar(x="Feature", y=["Out of Sample"], rot=0)
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.title("Feature Importance in Random Forest Model")
    plt.show()


def neural_network_feature_importance():
    # Set up data
    data = pd.read_csv("../data/custom_8.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with 0
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    data = data.dropna(subset=["USD_MSRP", "Current_Price"])  # Drop rows with missing data
    features = ["Year", "Pieces", "Theme", "Minifigures", "Rating", "Owned", "USD_MSRP"]
    clean = data[features]
    data = clean.to_numpy()

    # Rotate through features to test
    num_features = data.shape[1] - 1
    num_rows = data.shape[0]
    experiment_results = pd.DataFrame(columns=["Feature", "R2_IS", "R2_OS"])
    for i in range(num_features + 1):

        # If we are testing a feature, scramble it, else test normal data
        feature_data = data.copy()
        if i < num_features:
            feature_data[:, i] = feature_data[:, i][np.random.permutation(num_rows)]

        # Set up learner
        learner = MLPRegressor(hidden_layer_sizes=(35, 45, 55), activation='relu', alpha=0.0001, max_iter=4000)

        # Set up cross validation
        kf = KFold(n_splits=5, shuffle=True)

        # Train and evaluate for each fold
        r2_is_scores = []
        r2_os_scores = []
        for train_indices, test_indices in kf.split(feature_data):
            # Get split
            x_train, y_train = feature_data[train_indices, :-1], feature_data[train_indices, -1]
            x_test, y_test = feature_data[test_indices, :-1], feature_data[test_indices, -1]

            # Train and fit
            learner.fit(x_train, y_train)

            # Test in sample
            in_sample_predictions = learner.predict(x_train)
            r_squared_in_sample = r2_score(y_train, in_sample_predictions)
            r2_is_scores.append(r_squared_in_sample)

            # Test out of sample
            predictions = learner.predict(x_test)
            r_squared = r2_score(y_test, predictions)
            r2_os_scores.append(r_squared)

        # Compute the mean R^2 score
        mean_r2_is = np.mean(r2_is_scores)
        mean_r2_os = np.mean(r2_os_scores)
        feature_results = pd.DataFrame({"Feature": features[i], "R2_IS": mean_r2_is, "R2_OS": mean_r2_os}, index=[0])
        experiment_results = pd.concat([experiment_results, feature_results], ignore_index=True)
        print(f"Finished testing feature: '{features[i]}'")

    # Print results
    print(experiment_results)
    importances = experiment_results
    importances["In Sample"] = importances["R2_IS"][6] - importances["R2_IS"]
    importances["Out of Sample"] = importances["R2_OS"][6] - importances["R2_OS"]
    importances = importances[importances["Feature"] != "USD_MSRP"]
    importances.plot.bar(x="Feature", y=["Out of Sample"], rot=0)
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.title("Feature Importance in Neural Network")
    plt.show()


def main():
    #random_forest_feature_importance()
    neural_network_feature_importance()


if __name__ == "__main__":
    main()