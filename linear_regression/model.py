import numpy as np

class LinearRegression:

	def __init__(self, learning_rate=0.01, n_iterations=1000, solver="gradient_descent"):
		"""
		Linear Regression Model supporting both OLS closed-form and Gradient Descent techniques.
		parameters:
		learning_rate: float value
		n_iterations: int value
		solver: str
			'gradient_descent' or 'ols'
		"""
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self. solver = solver

		self.weights = None
		self.bias = None

	def fit(self, X_train, y_train):
		
		if self.solver == "ols":
			self._fit_ols(X_train, y_train)

		elif self.solver == "gradient_descent":
			self._fit_gradient_descent(X_train, y_train)

		else:
			raise ValueError("Solver must be 'ols' or 'gradient_descent'")

	def _fit_ols(self, X_train, y_train):
		pass

	def _fit_gradient_descent(self, X_train, y_train):
		pass
	
	def predict(self, X_test):
		return np.dot(X_test, self.weights) + self.bias