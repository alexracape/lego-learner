import numpy as np


class BootstrapLearner:

    def __init__(self, constituent, kwargs, bags=20):
        self.learner_type = constituent
        self.parameters = kwargs
        self.bags = bags
        self.learners = np.empty(bags, dtype=object)

    def train(self, x, y):
        num_rows = len(x)
        for i in range(self.bags):
            sample = np.random.randint(0, num_rows, size=num_rows)
            learner = self.learner_type(**self.parameters)
            learner.train(x[sample], y[sample])
            self.learners[i] = learner

    def test(self, x):

        # return predictions (estimates) for each row of x
        y = np.zeros((self.bags, len(x)))
        for i in range(self.bags):
            predictions = self.learners[i].test(x)
            y[i] = predictions
        return y.mean(axis=0)
