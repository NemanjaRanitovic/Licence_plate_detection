# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 20:46:07 2023

@author: Nemanja
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import cv2
import easyocr
import xml.etree.ElementTree as xet
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from PIL import Image
import pytesseract




#%%
def getImage(file):
    xmlName = xet.parse(file).getroot().find('filename').text
    imageName = os.path.join('./images2',xmlName)
    return imageName

path = glob('./images2/*.xml')

dataLabels = dict(path = [], x1 = [], x2 = [], y1 = [], y2 = [])
for dataXML in path:
        info = xet.parse(dataXML)
        coordinates = info.getroot().find('object').find('bndbox')
        x1 = int(coordinates.find('xmin').text)
        x2 = int(coordinates.find('xmax').text)
        y1 = int(coordinates.find('ymin').text)
        y2 = int(coordinates.find('ymax').text)
        dataLabels['path'].append(dataXML)
        dataLabels['x1'].append(x1)
        dataLabels['x2'].append(x2)
        dataLabels['y1'].append(y1)
        dataLabels['y2'].append(y2)

df = pd.DataFrame(dataLabels)
images = list(df['path'].apply(getImage))
labels = df[['x1','x2','y1','y2']].to_numpy()

data = []
output = []
for index in range(len(images)):
    
    #Loading array of normalised images
    image = images[index]
    imageArray = cv2.imread(image)
    h,w,d = imageArray.shape
    loadImage = load_img(image,target_size = (224,224))
    loadImageArray = img_to_array(loadImage)
    normalisedImageArray = loadImageArray/255.0
    
    #Loading normalised box coordinates
    x1,x2,y1,y2 = labels[index]
    nx1,nx2 = x1/w,x2/w
    ny1,ny2 = y1/h,y2/h
    normalisedCoord = (nx1,nx2,ny1,ny2)
    
    data.append(normalisedImageArray)
    output.append(normalisedCoord)
    
#%%
imagesSplit = np.array(data,dtype = np.float32)
xmlSplit = np.array(output,dtype = np.float32)

xtrain,xtest,ytrain,ytest = train_test_split(imagesSplit,xmlSplit,train_size = 0.8,random_state = 0)
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)

#%%
inceptionResnet = InceptionResNetV2(weights = "imagenet",include_top = False,
                                     input_tensor=Input(shape=(224,224,3)))
inceptionResnet.trainable = False
headmodel = inceptionResnet.output
headmodel = Flatten()(headmodel)


## ovde je mesto gde moze kod da se optimizuje
## dodavanjem dens sloja sa 1000 neurona pravi razliku kod nekih slika na bolje, kod nekih na losije. 

headmodel = Dense(500,activation = "relu")(headmodel)
headmodel = Dense(250,activation = "relu")(headmodel) 
headmodel = Dense(4,activation = 'sigmoid')(headmodel) 

model = Model(inputs = inceptionResnet.input, outputs = headmodel)

model.compile(loss = "mse",optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4))
model.summary()
tensorBoard = TensorBoard('licenceDetector')

#%%
model = load_model('./models/mreza.h5')
#%%
image = load_img('./test_images/Test15.png')
image = np.array(image,dtype=np.uint8)
image1 = load_img('./test_images/Test15.png',target_size = (224,224))
image_arr = img_to_array(image1)/255.0
h,w,d = image.shape
plt.figure(figsize = (10,8))

#Ako ne prikazemo sliku prvi put, slika ce biti jasnija, ako se prikazu dve slike jedna za drugom plot drugu zamuti
plt.imshow(image)
plt.show()

predicting_array = image_arr.reshape(1,224,224,3)
coordinates = model.predict(predicting_array)


true_values = np.array([w,w,h,h])

coordinates = coordinates * true_values

coordinates = coordinates.astype(np.int32)
x1,x2,y1,y2 = coordinates[0]

#%% Tessaract
leftcorner = (x1,y1)
rightcorner = (x2,y2)
cv2.rectangle(image,leftcorner,rightcorner,(0,255,0),1)
plt.figure(figsize = (10,8))
plt.imshow(image)
plt.show()

image_arr = array_to_img(image_arr)
crop_coordinates = (x1,y1,x2,y2)
croppedImage = array_to_img(image).crop(crop_coordinates)
plt.figure(figsize = (10,8))
plt.imshow(croppedImage)
plt.show()
text = pytesseract.image_to_string(croppedImage,lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
print(text)




#%% EASYOCR
reader = easyocr.Reader(['en'], gpu=False)
text = reader.readtext(img_to_array(croppedImage))
print(text)



## Ne snalazi se se zutim tablicama tj sa tablicama koje nisu bele
