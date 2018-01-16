import numpy as np
import cv2 # To read the images by pixel
import pandas as pd
import matplotlib.pyplot as plt
import glob # To read the location path
import os # To read the location path and add the extention
import random # To generate random index numbers

# train set: 19426 images
# valid set: 6523 images
# each image is 100x100 pixels
# there are 41 classes (fruit types)
# r_img_idx mean rotated fruit

fruit_images = []
labels = []
valid_img = []
valid_labels = []

for fruit_dir_path in glob.glob("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/fruits-360/Training/*"):
   fruit_label = fruit_dir_path.split("/")[-1]
   for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
       image = cv2.imread(image_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (30, 30))
       #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # why convert color?
       fruit_images.append(image)
       labels.append(fruit_label[9:])

for fruit_dir_path in glob.glob("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/fruits-360/Validation/*"):
   fruit_label = fruit_dir_path.split("/")[-1]
   for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
       image = cv2.imread(image_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (30, 30))
       #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # why convert color?
       valid_img.append(image)
       valid_labels.append(fruit_label[11:])

from sklearn.preprocessing import OneHotEncoder, LabelEncoder # To categorize the fruit labels
encoder = LabelEncoder()
encoder_int =encoder.fit_transform(labels) # turn the labels to 0 - 45
encoder_int = encoder_int.reshape(len(encoder_int), 1)
onehot_encoder = OneHotEncoder(sparse=False)
labels_encoded = onehot_encoder.fit_transform(encoder_int)

encoder = LabelEncoder()
valid_encoder_int =encoder.fit_transform(valid_labels) # turn the labels to 0 - 40
valid_encoder_int = valid_encoder_int.reshape(len(valid_encoder_int),1)
valid_onehot_encoder = OneHotEncoder(sparse=False)
valid_labels_encoded = valid_onehot_encoder.fit_transform(valid_encoder_int)

cv2.imshow(' ', fruit_images[100])
cv2.waitKey(1)

standardized_image = [] # to standardize each image: by dividing everything by 255
for image in fruit_images:
   standardized_image.append(image/255.)
valid_standardized_image = []
for image in valid_img:
   valid_standardized_image.append(image/255.)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(30, 30, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=2, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(41, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
standardized_image2 = []
for image in standardized_image:
   standardized_image2.append(image.tolist())
valid_standardized_image2 = []
for image in valid_standardized_image:
   valid_standardized_image2.append(image.tolist())
model.fit(standardized_image2,labels_encoded ,validation_data=(valid_standardized_image2, valid_labels_encoded )
         , batch_size=20,epochs = 5 )

# 1/12/2018
# try to merge training set and validation set
# randomly select N image from each category.
# randomly split into training (70%) and validation (20%) and test set(10%).
fruit_images = []
labels = []
for fruit_dir_path in glob.glob("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/fruits-360/Training/*"):
   fruit_label = fruit_dir_path.split("/")[-1]
   for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
       image = cv2.imread(image_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (30, 30))
       #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # why convert color?
       fruit_images.append(image)
       labels.append(fruit_label[9:])
for fruit_dir_path in glob.glob("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/fruits-360/Validation/*"):
   fruit_label = fruit_dir_path.split("/")[-1]
   for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
       image = cv2.imread(image_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (30, 30))
       #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # why convert color?
       fruit_images.append(image)
       labels.append(fruit_label[11:])
len(labels)

idx = random.sample(range(len(labels)),len(labels)) # randomize the index
idx = np.array(idx) # turn into array in order to make the calculation faster
labels = np.array(labels)
fruit_images = np.array(fruit_images)
int(len(idx)*0.7)#70% of the data
# frequency table
from collections import Counter
freq = Counter(labels)
a = ['a','b','c','d','b','c','a','d','e']
b = random.sample(range(len(a)),len(a)) # randomize the index
b = np.array(b)
a = np.array(a)
a = a[b].tolist()
a
