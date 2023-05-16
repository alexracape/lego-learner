
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():

    data = pd.read_csv("./data/custom_8.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data["Pieces"] = data["Pieces"].fillna(-1)  # Fill in missing piece data with -1, is this appropriate? need to test
    data["Theme"] = data["Theme"].astype('category').cat.codes  # Categorical code approach
    # counts = data["Theme"].value_counts().to_dict()  # Frequency encoding approach, some themes have same count tho
    # data["Theme"] = data["Theme"].map(counts)
    data = data.dropna()  # Drop rows with missing data - 1939 remaining...
    clean = data[["Year", "Pieces", "Minifigures", "Theme", "USD_MSRP"]]
    data = clean.to_numpy()
    num_features = 5


    # Normalize the input data
    scaler = StandardScaler()
    data[:, :-1] = scaler.fit_transform(data[:, :-1])


    # Set up the network
    mlp = MLPRegressor(hidden_layer_sizes=(35, 45, 55), activation='relu', alpha=0.0001, max_iter=4000)

    # Set up cross validation
    num_folds = 10
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
        mlp.fit(x_train, y_train)

        # Test in sample
        in_sample_predictions = mlp.predict(x_train)
        rmse_in_sample = mean_squared_error(y_train, in_sample_predictions, squared=False)
        correlation_in_sample = np.corrcoef(y_train.astype(np.float64), in_sample_predictions)[0, 1]
        rmse_in_sample_scores.append(rmse_in_sample)
        correlation_in_sample_scores.append(correlation_in_sample)

        # Test out of sample
        predictions = mlp.predict(x_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        correlation = np.corrcoef(y_test.astype(np.float64), predictions)[0, 1]
        rmse_scores.append(rmse)
        correlation_scores.append(correlation)

    # Compute the mean score over all folds
    mean_rmse_os = np.mean(rmse_scores)
    mean_correlation_os = np.mean(correlation_scores)
    mean_rmse_is = np.mean(rmse_in_sample_scores)
    mean_correlation_is = np.mean(correlation_in_sample_scores)

    # Print Results
    # print(f"Mean Squared Error (in sample): {mean_rmse_is}\n\t{rmse_in_sample_scores}")
    # print(f"Correlation (in sample): {mean_correlation_is}\n\t{correlation_in_sample_scores}")
    print(f"Mean Squared Error: {mean_rmse_os}\n\t{rmse_scores}")
    # print(f"Correlation: {mean_correlation_os}\n\t{correlation_scores}")

    print(f"Correlation (in sample): {mean_correlation_is}")
    print(f"Correlation (out sample): {mean_correlation_os}")

    print(f"Mean Squared Error: {mean_rmse_os}")
    print(f"Mean Squared Error (in sample): {mean_rmse_is}")

    return mean_correlation_is, mean_correlation_os

    # So far correlations are around .5, .64 at best need to tune hyperparameters, features, and model structure


if __name__ == "__main__":
    # Set up empty lists to store results
    # mean_correlation_is_list = []
    # mean_correlation_os_list = []

    # # Run main() 100 times with different values of counter
    # counter = 0
    # while counter < 10000:
    #     # print(f"Counter: {(counter+1, counter+10, counter+20)}")
    #     print(f"Interations: {1000+counter}")
    #     mean_correlation_is, mean_correlation_os = main(counter)
    #     mean_correlation_is_list.append(mean_correlation_is)
    #     mean_correlation_os_list.append(mean_correlation_os)
    #     counter += 1000

    # # Create dataframe from results
    # df = pd.DataFrame({'mean_correlation_is': mean_correlation_is_list, 'mean_correlation_os': mean_correlation_os_list})
    
    
    
    # Plot results
    # df.plot()
    # plt.xlabel('Counter')
    # plt.ylabel('Correlation')
    # plt.title('MLPRegressor Performance')
    # plt.show()
    
    main()
