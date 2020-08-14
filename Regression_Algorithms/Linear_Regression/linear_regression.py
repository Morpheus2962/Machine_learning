import numpy as np
from tqdm import tqdm

class Linear_Regression:

	def __init__(self, lr = 0.01, epochs = 1000):

		self.lr = lr 		      # Learning Rate
		self.epochs = epochs      # No. of Epochs  
		self.weights = None		  # weights
		self.bias = None  	  # bias


	def fit(self, X, y):

		samples_n, inputs_n = X.shape

		#samples_n = no. of samples
		#inputs_n = no. of inputs per sample

		self.weights = np.zeros(inputs_n)   # make array of weights filled with zeros of shape (inputs_n)
		self.bias = 0						# set bias as 0

		for epoch in tqdm(range(self.epochs)):  # iterate through all epochs
  			# y_1 is predicted value 
			y_1 = np.dot(X, self.weights) + self.bias 


			# here we calculate gradients of MSE(cost function) to use gradient descent to minimize it
			''' 
				MSE = Mean Squared Error or L2 loss
				Formula of MSE is:
					MSE=1/N ( ∑i=1 to n {(y[i]−(m * x[i] + b))**2} )
					N = no of sample
					y = orignal value
					m = weights
					b = bias
					(m * x + b) = predicted value
			'''

			# dw = derivative of weight,  db = derivative of bias
			dw = (1/samples_n) * np.dot(X.T, (y_1 - y))
			db = (1/samples_n) * np.sum(y_1 - y)

			# here we update parameters to reduce MSE

			self.weights -= self.lr * dw
			self.bias  -= self.lr * db

	def predict(self, X_val):
		# here we predict data using our updated weights and bias

		predicted_val = np.dot(X_val, self.weights) + self.bias

		return predicted_val


