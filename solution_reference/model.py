"""
Reference Solution: Linear Regression with Gradient Descent
Instructor Use Only
"""

import numpy as np


class LinearRegressionGD:
    """Linear Regression using Gradient Descent optimization."""

    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        Initialize the model.

        Args:
            learning_rate: Step size for gradient descent (alpha)
            max_iterations: Maximum number of training iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.slope = 0.0
        self.intercept = 0.0
        self.history = {
            'iteration': [],
            'cost': [],
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': []
        }

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Args:
            X: Feature array (study hours)
            y: Target array (exam scores)

        Returns:
            self (for method chaining)
        """
        # Ensure arrays are flattened
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        n = len(X)

        # Initialize parameters
        m = 0.0  # slope
        b = 0.0  # intercept

        # Gradient descent loop
        for iteration in range(self.max_iterations):
            # Forward pass: make predictions
            y_pred = m * X + b

            # Compute cost (MSE/2)
            cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2)

            # Compute gradients (partial derivatives)
            grad_m = (1 / n) * np.sum((y_pred - y) * X)
            grad_b = (1 / n) * np.sum(y_pred - y)

            # Update parameters
            m = m - self.learning_rate * grad_m
            b = b - self.learning_rate * grad_b

            # Store history
            self.history['iteration'].append(iteration)
            self.history['cost'].append(cost)
            self.history['slope'].append(m)
            self.history['intercept'].append(b)
            self.history['grad_slope'].append(grad_m)
            self.history['grad_intercept'].append(grad_b)

        # Store final parameters
        self.slope = m
        self.intercept = b

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Feature array (study hours)

        Returns:
            Predicted values (exam scores)
        """
        X = np.array(X).flatten()
        return self.slope * X + self.intercept

    def calculate_metrics(self, X, y):
        """
        Calculate performance metrics.

        Args:
            X: Feature array
            y: True target values

        Returns:
            Dictionary with metrics: R², MSE, RMSE, MAE
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        # R² (coefficient of determination)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - y.mean()) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)

        # MSE (Mean Squared Error)
        mse = np.mean((y - y_pred) ** 2)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y - y_pred))

        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def get_parameters(self):
        """
        Get the trained model parameters.

        Returns:
            Dictionary with slope and intercept
        """
        return {
            'slope': self.slope,
            'intercept': self.intercept
        }

    def get_history(self):
        """
        Get the training history.

        Returns:
            Dictionary with training history
        """
        return self.history

    def calculate_residuals(self, X, y):
        """
        Calculate residuals (prediction errors).

        Args:
            X: Feature array
            y: True target values

        Returns:
            Array of residuals
        """
        y_pred = self.predict(X)
        return y - y_pred
