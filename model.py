import numpy as np


class LinearRegressionGD:
    """Linear Regression using (optionally weighted) Gradient Descent optimization."""

    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.slope = 0.0
        self.intercept = 0.0
        self.history = {
            "iteration": [],
            "cost": [],
            "slope": [],
            "intercept": [],
            "grad_slope": [],
            "grad_intercept": []
        }

    def fit(self, X, y, sample_weights=None, sample_weight=None, **kwargs):
        """
        Train the model using gradient descent.

        Supports optional weights:
          - sample_weights: array-like shape (n,)
          - sample_weight: alias

        Uses weighted MSE cost:
          J = (sum w_i * (yhat_i - y_i)^2) / (2 * sum w_i)
        """
        X = np.asarray(X, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        n = len(X)

        # Accept either name
        if sample_weights is None and sample_weight is not None:
            sample_weights = sample_weight

        # Build weights
        if sample_weights is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weights, dtype=float).flatten()
            if len(w) != n:
                raise ValueError(f"sample_weights length {len(w)} must match X/y length {n}")
            w = np.clip(w, 1e-12, np.inf)  # safety

        w_sum = float(np.sum(w))

        m = 0.0
        b = 0.0

        # Reset history each fit so plots match current run
        self.history = {
            "iteration": [],
            "cost": [],
            "slope": [],
            "intercept": [],
            "grad_slope": [],
            "grad_intercept": []
        }

        for iteration in range(self.max_iterations):
            y_pred = m * X + b
            err = y_pred - y

            # Weighted cost
            cost = (1.0 / (2.0 * w_sum)) * np.sum(w * (err ** 2))

            # Weighted gradients
            grad_m = (1.0 / w_sum) * np.sum(w * err * X)
            grad_b = (1.0 / w_sum) * np.sum(w * err)

            # Update
            m -= self.learning_rate * grad_m
            b -= self.learning_rate * grad_b

            # Save history
            self.history["iteration"].append(iteration)
            self.history["cost"].append(cost)
            self.history["slope"].append(m)
            self.history["intercept"].append(b)
            self.history["grad_slope"].append(grad_m)
            self.history["grad_intercept"].append(grad_b)

        self.slope = float(m)
        self.intercept = float(b)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).flatten()
        return self.slope * X + self.intercept

    def calculate_metrics(self, X, y, sample_weights=None, sample_weight=None, **kwargs):
        """Calculate weighted (or unweighted) R², MSE, RMSE, MAE."""
        X = np.asarray(X, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        y_pred = self.predict(X)

        if sample_weights is None and sample_weight is not None:
            sample_weights = sample_weight

        if sample_weights is None:
            w = np.ones_like(y, dtype=float)
        else:
            w = np.asarray(sample_weights, dtype=float).flatten()
            if len(w) != len(y):
                raise ValueError(f"sample_weights length {len(w)} must match y length {len(y)}")
            w = np.clip(w, 1e-12, np.inf)

        w_sum = float(np.sum(w))

        # Weighted mean of y for R²
        y_bar = float(np.sum(w * y) / w_sum)

        ss_res = float(np.sum(w * ((y - y_pred) ** 2)))
        ss_tot = float(np.sum(w * ((y - y_bar) ** 2)))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        mse = ss_res / w_sum
        rmse = float(np.sqrt(mse))
        mae = float(np.sum(w * np.abs(y - y_pred)) / w_sum)

        return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

    def get_parameters(self):
        return {"slope": self.slope, "intercept": self.intercept}

    def get_history(self):
        return self.history

    def calculate_residuals(self, X, y):
        y = np.asarray(y, dtype=float).flatten()
        y_pred = self.predict(X)
        return y - y_pred
