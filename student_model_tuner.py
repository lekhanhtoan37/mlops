from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd

class StudentModelTuner:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestRegressor()
        self.best_params = None

    def train_and_tune(self):
        X = self.data.drop("score", axis=1)
        y = self.data["score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Best Parameters: {self.best_params}")
        print(f"Mean Squared Error: {mse}")
        return best_model
