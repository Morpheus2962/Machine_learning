import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# import our model 
from linear_regression import Linear_Regression

# function to find Mean Squared Error
def MSE(y_val, y_pred):
    return np.mean((y_val - y_pred)**2)

# data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# split data in training data and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.26, random_state=1234)

# load model
model = Linear_Regression(lr = 0.01, epochs = 1000)

# Train model
model.fit(X_train, y_train)

# save predictions 
predictions = model.predict(X_val)


print(f"MSE = {MSE(predictions, y_val)}")


#plot the data 
pred_line = model.predict(X)
cmap = plt.get_cmap('magma')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.85), s=40)
m2 = plt.scatter(X_val, y_val, color=cmap(0.35), s=40)
plt.plot(X, pred_line, color='black', linewidth=2, label="Prediction")
plt.text(0.63, -200, "Orange dots = training data\nPurple dots = validation data", bbox=dict(facecolor='white', alpha=0.8))
plt.show()





