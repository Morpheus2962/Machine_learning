

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D
import pickle
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

 
NAME = "CatsandDogsNN-OPt-{}".format(int(time.time()))
#LOAD OUR TRAINING DATA
X = pickle.load(open("X.pickle", "rb"))   
y = pickle.load(open("y.pickle", "rb"))


X = np.array(X/255.0) 
y = np.array(y)

#LAYER1
model = Sequential()  # it is a sequential model
model.add(Conv2D((32),(3,3), input_shape = X.shape[1:])) # add convolutional layer
model.add(Activation("relu")) # add activation funtion (relu = Rectified Linear Unit)
model.add(MaxPooling2D(pool_size=(2,2))) # downsample our data
model.add(Dropout(rate = 0.25))

#LAYER2
model.add(Conv2D((32),(3,3) ))   
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.2))
#LAYER3
model.add(Conv2D((64),(3,3) ))   
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.3))
#LAYER4
model.add(Conv2D((128),(3,3) ))   
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.4))

#LAYER5
model.add(Flatten()) 
model.add(Dense(128)) 
model.add(Activation("relu"))

#OUTPUT LAYER
model.add(Dense(1))
model.add(Activation('sigmoid'))  

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


#COMPILE THE MODEL i.e. CONFIGURE THE MODEL
model.compile(loss='binary_crossentropy',  
             optimizer='adam',		
             metrics=['accuracy'])  
 
#TRAIN OUR MODEL
model.fit(X,y, batch_size = 32, epochs = 10, validation_split=0.3, callbacks=[tensorboard] )  
  
model.save('CAD_CNN_Opt.model') # save our model
