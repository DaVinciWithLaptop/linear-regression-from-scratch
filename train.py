import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from linear_regression.model import LinearRegression
from utils.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

# Generating synthetic dataset
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training using OLS closed-form
ols_model = LinearRegression(solver="ols")
ols_model.fit(X_train, y_train)

ols_predictions = ols_model.predict(X_test)

print("\n---------- OLS Results ----------")
print("\nMSE:", mean_squared_error(y_test, ols_predictions))
print("\nRMSE:", root_mean_squared_error(y_test, ols_predictions))
print("\nMAE:", mean_absolute_error(y_test, ols_predictions))
print("\nR2:", r2_score(y_test, ols_predictions))

# Training using Gradient Descent
gd_model = LinearRegression(solver="gradient_descent", learning_rate=0.01, n_iterations=1000)
gd_model.fit(X_train, y_train)

gd_predictions = gd_model.predict(X_test)

print("\n---------- Gradient Descent Results ----------")
print("\nMSE:", mean_squared_error(y_test, gd_predictions))
print("\nRMSE:", root_mean_squared_error(y_test, gd_predictions))
print("\nMAE:", mean_absolute_error(y_test, gd_predictions))
print("\nR2:", r2_score(y_test, gd_predictions))


# OLS Visualization
plt.figure(figsize=(8,6))

plt.scatter(y_test, ols_predictions, alpha=0.7)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("OLS: Actual vs Predicted")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")

plt.show()


# Gradient Descent Visualization
plt.figure(figsize=(8,6))

plt.scatter(y_test, gd_predictions, alpha=0.7)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Gradient Descent: Actual vs Predicted")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="orange")

plt.show()