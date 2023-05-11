
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np


def main():

    # Set up data
    data = pd.read_csv("../data/custom_6.csv", delimiter=",", quotechar='"')
    data["Minifigures"] = data["Minifigures"].fillna(0)  # Fill in missing minifigure data with 0
    data = data.dropna()  # Drop rows with missing data - 1939 remaining...
    clean = data[["Year", "Pieces", "Minifigures", "USD_MSRP", "Current_Price"]]  # Removed theme and subtheme for now
    data = clean.to_numpy()
    num_features = 4

    # Shuffle the rows and partition some data for testing.
    features = data[:, :-1]
    labels = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    # Set up the network
    mlp = MLPRegressor(hidden_layer_sizes=(num_features, num_features, num_features), activation='relu', max_iter=1000)
    mlp.fit(x_train, y_train)

    # Evaluate the network
    predictions = mlp.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    correlation = np.corrcoef(y_test.astype(np.float64), predictions)[0, 1]
    print(f"Mean Squared Error: {rmse}")
    print(f"Correlation: {correlation}")
    # So far correlations are around .5, .57 at best


if __name__ == "__main__":
    main()
