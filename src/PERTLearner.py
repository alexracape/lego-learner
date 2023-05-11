import numpy as np


class PERTLearner:

    def __init__(self):
        self.feature = None
        self.split_val = None
        self.y_val = None
        self.left = None
        self.right = None

    def train(self, x, y):

        num_rows, num_features = x.shape
        a = b = 0
        tries = 0
        while a == b:

            # Select random feature
            self.feature = np.random.randint(0, num_features - 1)

            # Select 2 random rows and get value at feature
            w, z = np.random.randint(0, num_rows, size=2)
            a, b = x[w, self.feature], x[z, self.feature]
            tries += 1

            # If unable to find valid split after 10 tries, return leaf
            if tries == 10:
                leaf = PERTLearner()
                leaf.y_val = np.mean(y)
                return leaf

        # Get split val from valid a and b values
        self.split_val = (.5 * a) + (.5 * b)

        # Recurse with child leafs using a mask to split the data
        feature_col = x[:, self.feature]
        mask = feature_col <= self.split_val
        left = PERTLearner()
        right = PERTLearner()
        self.left = left.train(x[mask], y[mask])
        self.right = right.train(x[~mask], y[~mask])
        return self

    def query(self, x):

        if self.y_val:
            return self.y_val

        if x[self.feature] <= self.split_val:
            return self.left.query(x)
        else:
            return self.right.query(x)

    def test(self, x):

        # return predictions (estimates) for each row of x
        num_rows = len(x)
        y = np.zeros(num_rows)
        for i in range(num_rows):
            pred = self.query(x[i])
            y[i] = pred
        return y

    def __repr__(self) -> str:
        if self.y_val:
            return f"Leaf Node with val = {self.y_val}"
        else:
            return f"Branch Feature: {self.feature} @ {self.split_val}\n\t{self.left}\n\t{self.right}"
