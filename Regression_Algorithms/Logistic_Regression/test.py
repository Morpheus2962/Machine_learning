import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# import our model 
from logistic_regression import Logistic_Regression

# function to find accuracy of model
def accuracy(y_val, y_pred):

    correct = np.sum(y_val == y_pred)
    total = len(y_val)
    accuracy = correct/total * 100
    return round(accuracy, 4)

# load data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target


# split data in training data and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)

# load model
model = Logistic_Regression(lr=0.0001, epochs=1000)
# Train model
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_val)

print(f"Accuracy : {accuracy(y_val, predictions)} %")
