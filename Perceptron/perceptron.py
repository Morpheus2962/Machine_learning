import numpy as np
from tqdm import tqdm
#Make our Perceptron Model
class Perceptron:

    def __init__(self, lr=0.01, epoch=1000):
        self.lr = lr
        self.epoch = epoch
        self.activation_func = self.unit_step_function  #set activation function
        self.weights = None
        self.bias = None
    # This function is to train our perceptron
    def fit(self, X, y):   
        samples_n, features_n = X.shape
        #samples_n =   # no. of samples
        #features_n =  # no. of features(labels) 

        # init parameters
        self.weights = np.zeros(features_n)  # make weights eqaul to number of inputs
        self.bias = 0

        y1 = np.array([1 if i > 0 else 0 for i in y])  #sets value of y as 0 or 1

        for e in tqdm(range(self.epoch)):
            
            for index, X_input in enumerate(X):

                linear_output = np.dot(X_input, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y1[index] - y_predicted)

                self.weights += update * X_input
                self.bias += update
    # This function is to predict data
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


    # Activation function
    def unit_step_function(self, x):
        '''
            np.where(condition,a,b)
                here condition is x>0
                and a = 1
                    b = 0

                if condition (i.e. x>0 ) is true:
                    return a
                else 
                    return b
        '''
        return np.where(x>=0, 1, 0)  




