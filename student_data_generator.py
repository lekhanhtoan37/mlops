import numpy as np
from sklearn.datasets import make_regression
import pandas as pd

class StudentDataGenerator:
    def __init__(self, n_samples=1000, n_features=5, noise=0.1, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.random_state = random_state

    def generate(self):
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
            random_state=self.random_state
        )
        columns = [f"feature_{i}" for i in range(self.n_features)]
        data = pd.DataFrame(X, columns=columns)
        data["score"] = y
        return data
