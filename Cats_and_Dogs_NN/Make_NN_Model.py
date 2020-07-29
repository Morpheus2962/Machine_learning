##TO MAKE NEURAL NETWORK MODEL HERE I HAVE USED TENSORFLOW LIBRARY

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D
import pickle
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

 
#LOAD OUR TRAINING DATA
X = pickle.load(open("X.pickle", "rb"))   
y = pickle.load(open("y.pickle", "rb"))


X = np.array(X/255.0) #normalize(scale) our data (as data is image max value can be 255 and minimum be 0. So we divide by 255 so it is between range 0-1)
y = np.array(y)

##HERE WE MAKE OUR MODEL


#LAYER1
model = Sequential()  # it is a sequential model
model.add(Conv2D((64),(3,3), input_shape = X.shape[1:])) # add convolutional layer
model.add(Activation("relu")) # add activation funtion (relu = Rectified Linear Unit)
model.add(MaxPooling2D(pool_size=(2,2))) # downsample our data


#LAYER2
model.add(Conv2D((64),(3,3), input_shape = X.shape[1:]))   
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

#LAYER#
model.add(Flatten()) #flattens our data e.g. [[1,2,3],[4,5,6]] to [[1,2,3,4,5,6]]
model.add(Dense(64)) # add dense layer 64 is dimension of layer(64 nodes)
model.add(Activation("relu"))

#OUTPUT LAYER
model.add(Dense(1))
model.add(Activation('sigmoid')) # add sigmoid activation function 


#COMPILE THE MODEL i.e. CONFIGURE THE MODEL
model.compile(loss='binary_crossentropy',  # set loss function
             optimizer='adam',		# set optimizer
             metrics=['accuracy'])  # set metrics to be evaluated 
 
#TRAIN OUR MODEL
model.fit(X,y, batch_size = 32, epochs = 3, validation_split=0.1, )  
# batch size is no of samples.  
#epochs is number of iteration over our data.  
#validation split is % of training data to be used as validation data
