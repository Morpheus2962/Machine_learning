import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from perceptron import Perceptron
#from pr import Perceptron
'''
	Here we make our training data using Sklearn library
	The function make_blobs() makes training data
	n_samples = no. of training samples
	n_features = no. of labels for training sample
	centers  = no. of classes betn features i.e. 0 and 1 
	'''
X , y =datasets.make_blobs(n_samples = 150, n_features = 2 , centers = 2,cluster_std=1.05,random_state=2 )
'''
	Here we use train_test_split() function to split our data in :
		1) X_train , y_train  i.e.  training data (used to train our data)
		2) X_test , y_test  i.e.  testing our data (to check accuracy)
	test_size = % to split our data (0.2 i.e. 20%)
'''
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state = 123)


#set our model as perceptron model
model = Perceptron()
#Train our model
model.fit(X_train,y_train)
#make array of all predictions using test data
predictions = model.predict(X_test)

#here we make accuracy function to check accuracy of our model
def accuracy(true , pred):
	corr = 0
	for i in range(len(pred)):
		if true[i] == pred[i]:
			corr+=1
	acc = corr/len(pred)*100
	return acc


print(f"Accuracy : {accuracy(y_test,predictions)}%" )

#predict a random sample 
pr = model.predict(X_test[2])
print(pr)
print(model.weights)



#plot data 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1],marker='o',c=y_train)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]

ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([ymin-3,ymax+3])

plt.show()

