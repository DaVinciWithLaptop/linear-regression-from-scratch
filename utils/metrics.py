import numpy as np

def mean_squared_error(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)


def root_mean_squared_error(y_true, y_pred):
	return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true, y_pred):
	return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
	
	ss_residual = np.sum((y_true - y_pred) ** 2)
	ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

	return 1 - (ss_residual / ss_total) 