import numpy as np
from tqdm import tqdm

class Logistic_Regression:

    def __init__(self, lr = 0.01, epochs = 1000):

        self.lr = lr              # Learning Rate
        self.epochs = epochs      # No. of Epochs  
        self.weights = None       # weights
        self.bias = None      # bias


    def fit(self, X, y):

        samples_n, inputs_n = X.shape

        #samples_n = no. of samples
        #inputs_n = no. of inputs per sample

        self.weights = np.zeros(inputs_n)   # make array of weights filled with zeros of shape (inputs_n)
        self.bias = 0                       # set bias as 0

        for epoch in tqdm(range(self.epochs)):  # iterate through all epochs
            # y_1 is predicted value 
            y_1 = np.dot(X, self.weights) + self.bias 
            # here we use sigmoid funtion so change in weights and bias will be low
            y_sig = self.sigmoid(y_1)


            # here we calculate gradients of MSE(cost function) to use gradient descent to minimize it
            ''' 
                MSE = Mean Squared Error or L2 loss
                Formula of MSE is:
                    MSE=1/N ( ∑i=1 to n{(y[i]−(m * x[i] + b))**2} )
                    N = no of sample
                    y = orignal value
                    m = weights
                    b = bias
                    (m * x + b) = predicted value
            '''

            # dw = derivative of weight,  db = derivative of bias
            dw = (1/samples_n) * np.dot(X.T, (y_sig - y))
            db = (1/samples_n) * np.sum(y_sig - y)

            # here we update parameters to reduce MSE

            self.weights -= self.lr * dw
            self.bias  -= self.lr * db

    def predict(self, X):
        # here we predict data using our updated weights and bias
        y1 = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(y1)
        '''as we have use sigmoid funtion it retruns value between 0 and 1
           but we cant use it so we convert it to either 0 or 1
           we apply condition 1 if > 0.5
                              0 if <= 0.5
        '''
        y_predicted1 = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted1)
        

    def sigmoid(self, x):
        ''' 

            Sigmoid Function:
                S(x)= 1 / (1 + exponent(-x))
            returns value betn 0 to 1
        '''
        return 1/(1+np.exp(-x))


