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
		# n_samples = no. of samples in X_train
		n_samples = X_train.shape[0]

		# Adding the bias column :
		# X_augmented vector = [1 x1 x2 x3 ... xn]
		X_augmented = np.c_[np.ones((n_samples,1)), X_train]

		# closed-formed formula :
		# weights vector = [bias w1 w2 w3 ... wn]
		weights = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y_train

		#separate bias(intercept aka Beta_0) and weigths(coeficients aka Beta_n values)
		self.bias = weights[0]
		self.weights = weights[1:]

	def _fit_gradient_descent(self, X_train, y_train):
		
		n_samples, n_features = X_train.shape

		# initializing parameters
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iterations):

			# making predictions
			y_pred = np.dot(X_train,self.weights) + self.bias

			# calculate loss function gradients
			dw = (2 / n_samples) * np.dot(X_train.T,(y_pred - y_train))
			dw = (2 / n_samples) * np.sum(y_pred - y_train)

			# updating weights, rule: W_new <= W_old - aplha * dj/dw
			self.weights = self.weights - self.learning_rate * dw

			# updating bias, rule: bias_new <= bias_old - aplha * dj/db
			self.bias = self.bias - self.learning_rate * db
	
	def predict(self, X_test):
		return np.dot(X_test, self.weights) + self.bias