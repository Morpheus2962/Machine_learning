import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
from tqdm import tqdm 
DATADIR = "C:/Users/abc/Documents/py/Py_Projects/Samples"   #set path of data to be predicted
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100
predic_samples=[]
label = []
def prepare():
    for category in CATEGORIES:  # iterate over categories dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # index categories (0 or a 1). 0=Dog 1=Cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per category i.e. Dog and Cat
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))# resize to normalize data size
                a = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  
                predic_samples.append(a) 
                label.append(category) # add each image DATA with their respective LABEL to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
          

prepare()




model = tf.keras.models.load_model("CAD_CNN_Opt.model")


correct = 0
total = 20

for (i,j) in zip(predic_samples,label):
	pred = model.predict(i)
	print(f"Original label = {j}")
	print(f"Predicted label = {CATEGORIES[int(pred[0][0])]}")
	k = CATEGORIES[int(pred[0][0])]
	if j == k:
		correct +=1
	

print(f"correct predictions = {correct}")
print(f"wrong predictions = {total-correct}")
per = (correct/total)*100
print(f"Accuracy = {per}%")
