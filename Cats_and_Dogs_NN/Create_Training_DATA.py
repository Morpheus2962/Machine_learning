

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
from tqdm import tqdm         

DATADIR = "C:/Users/abc/Documents/py/kagglecatsanddogs_3367a/PetImages"   #set path of data

CATEGORIES = ["Dog", "Cat"]   #catagorize data



##THIS SECTION IS TO CONVERT IMAGE TO ARRAY 
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert image to array
        #plt.imshow(img_array, cmap='gray')  # graph it
       # plt.show()        #show graph
        break
    break




IMG_SIZE = 100    #size of pixels as in 50x50
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) #resize image by img_size i.e. to 50x50
#plt.imshow(new_array, cmap='gray')
#plt.show()




training_data=[]  # create an array to store our training data

##THIS FUNCTION IS TO RESIZE IMAGEs AND ADD IT TO trainind_data ARRAY
##TO CREATE OUR TRAINING DATA

def create_training_data():
    for category in CATEGORIES:  # iterate over categories dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # index categories (0 or a 1). 0=Dog 1=Cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per category i.e. Dog and Cat
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add each image DATA with their respective LABEL to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()  # run the function which will create training data

print(len(training_data))  #Check Length of training data


import random
random.shuffle(training_data)   #shuffle aur data to get randomized data
 

#for sample in training_data[:10]:
    #print(sample[1])  # to check data

X = []   # Data
y = []   # Labels of data


for features,labels in training_data:
    X.append(features) # add data in array X
    y.append(labels)	#add label of data in array y

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,1)   # covert X to numpy array and reshape it to use it for training
                                              # 1 is due to grayscale to make it simple 



import pickle

pickle_out= open("X.pickle","wb") #Set name of data and wb as to write func
pickle.dump(X, pickle_out)  #dump X in picle_out list
pickle_out.close()

pickle_out= open("y.pickle","wb") #for y 
pickle.dump(y, pickle_out)
pickle_out.close()



