import numpy as np
import cv2
import glob
import os


# train set: 21876 images
# valid set: 7353 images
# each image is 100x100 pixels
# there are 46 classes (fruit types)
# r_img_idx mean rotated fruit
# to display an image
# cv2.imshow(' ',fruit_images[100])
# cv2.waitKey(1)

def reading_images(fruit_img, fruit_labels, img_folder):
    for fruit_dir_path in glob.glob(img_folder):
        fruit_name = fruit_dir_path.split("\\")[-1]
#        print(fruit_name)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#            print(image_path)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpeg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#            print(image_path)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#            print(image_path)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
    return fruit_img, fruit_labels


# glob.glob("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/food/training/*")

fruit_images = []
labels = []

fruit_images, labels = reading_images(fruit_images, labels,
                                      "C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*")

len(fruit_images)
110 + 115 + 108 + 120 + 100 + 99 + 100 + 108 + 101 + 100 + 120 + 117 + 116 + 100 + 101 + 100 + 115 + 100 + 100 + 119 + 100 + 100

import collections
collections.Counter(labels)

def helper_label(list_of_labels):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    encoder = LabelEncoder()
    # turn the labels to 0 - 45
    encoder_int = encoder.fit_transform(list_of_labels)
    encoder_int = encoder_int.reshape(len(encoder_int), 1)
    one_hot_encoder = OneHotEncoder(sparse=False)
    labels_encoded = one_hot_encoder.fit_transform(encoder_int)
    return labels_encoded

encoded_labels = helper_label(labels)

# generate random index numbers
import random

# randomize the index
idx = random.sample(range(len(labels)), len(labels))

# turn into array, to make the calculation faster?
idx = np.array(idx)
encoded_labels = np.array(encoded_labels)
fruit_images = np.array(fruit_images)
# save those indexes
train_idx = idx[:int(len(idx) * 0.8)]
validation_idx = idx[int(len(idx) * 0.8):int(len(idx) * 0.95)]
test_idx = idx[int(len(idx) * 0.95):]

# apply those index to training, validation, and test set
labels_train = encoded_labels[train_idx]
image_train = fruit_images[train_idx]
labels_valid = encoded_labels[validation_idx]
image_valid = fruit_images[validation_idx]
labels_test = encoded_labels[test_idx]
image_test = fruit_images[test_idx]
# check the distribution of the labels
sum(encoded_labels)
sum(labels_train)
sum(labels_valid)
sum(labels_test)
# frequency table, if needed
# from collections import Counter
# freq = Counter(labels)



# run a benchmark model
# basic benchmark model (in development)
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential

benchmark_model = Sequential()
benchmark_model.add(Flatten(input_shape=image_train.shape[1:]))  # image_train.shape[1:] is (100,100,3)
benchmark_model.add(Dense(256, activation='relu'))
# benchmark_model.add(Dropout(0.3))
benchmark_model.add(Dense(256, activation='relu'))
benchmark_model.add(Dropout(0.3))
benchmark_model.add(Dense(22, activation='softmax'))
benchmark_model.summary()

benchmark_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
benchmark_model.fit(image_train, labels_train,
                    validation_data=(image_valid, labels_valid),
                    batch_size=50, epochs=10)

# train the model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping


def cnn_from_scratch():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(22, activation='softmax'))
    model.summary()
    return model

# optimizer_generator generate an optimizer
# optimizer: "rms" or "sgd"
# learn_rate: default to 0.01
def optimizer_generator(optimizer, learn_rate=0.01):
    if optimizer not in ("rms", "sgd"):
        return "please enter rms or sgd as optimizer"
    if optimizer == "rms":
        return optimizers.RMSprop(lr=learn_rate)  # or 0.001 etc
    else:
        return optimizers.SGD(lr=learn_rate)

def fine_tuning(cnn_model, optimizer, learn_rate):

    model_optimizer = optimizer_generator(optimizer, learn_rate)
    cnn_model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    checkpt = ModelCheckpoint("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5", monitor='val_acc',
                              save_best_only=True, save_weights_only=False,
                              mode='auto', period=1)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, mode='auto')

    cnn_model.fit(image_train, labels_train,
                  validation_data=(image_valid, labels_valid),
                  batch_size=100, epochs=50, callbacks=[checkpt, early_stop])
    cnn_model.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")
    valid_pred_class = cnn_model.predict_classes(image_valid)
    number_correct = 0
    for i in range(len(labels_valid)):
        if valid_pred_class[i] == np.where(labels_valid == 1)[1][i]:
            number_correct += 1
    print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate, number_correct / len(valid_pred_class) * 100))

fine_tuning(cnn_from_scratch(), "rms", 0.1)

m = cnn_from_scratch()
m.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")
valid_pred_class = m.predict_classes(image_valid)
number_correct = 0
for i in range(len(labels_valid)):
    if valid_pred_class[i] == np.where(labels_valid == 1)[1][i]:
        number_correct += 1

# --------------------------transfer learning
from keras import applications
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(100, 100, 3))

for layer in model.layers[:]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)  # or 512
x = Dropout(0.2)(x)
x = Dense(22, activation='softmax')(x)

transfered_model = Model(inputs=model.input, output=x)
transfered_model.summary()

fine_tuning(transfered_model, "rms", 0.01)


transfered_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

checkpt = ModelCheckpoint("vgg16_1.h5", monitor='val_acc',
                          save_best_only=True, save_weights_only=False,
                          mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, mode='auto')

transfered_model.fit(image_train, labels_train,
                     validation_data=(image_valid, labels_valid),
                     batch_size=100, epochs=20, callbacks=[checkpt, early_stop])

# ------------------------end transfer learning --------------------------


# test the trained model on test set
# test_result: real label, number of position where 1 occurs (from 0-40)
# np.where(labels_test == 1): pred label, number of position where 1 occurs (from 0-40)
test_result = model.predict_classes(image_test)
number_correct = 0
for i in range(len(test_result)):
    if test_result[i] == np.where(labels_test == 1)[1][i]:
        number_correct += 1
print("The accuracy on test set is {:.2f}%".format(number_correct / len(test_result) * 100))

# test on real life picture
test_path = glob.glob("C:/Users/tam9/Desktop/Python/fruit_image/test/*")
# test_path = "C:/Users/tam9/Desktop/Python/fruit_image/test\lemon.jpg)"
test_img = cv2.imread(test_path[8], cv2.IMREAD_COLOR)
test_img = cv2.resize(test_img, (50, 50))
# test_img = test_img / 255.
test_img = test_img.reshape(1, 50, 50, 3)
model.predict_classes(test_img)

"C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*"

a = glob.glob(os.path.join("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*", "*.jpg"))
glob.glob(os.path.join("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*", "*.jpeg"))
a = glob.glob(os.path.join("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*", "*.png"))

a[1]

b = cv2.imread(a[1], cv2.IMREAD_COLOR)
c = cv2.resize(b, (50, 50))
len(c)

glob.glob(os.path.join(fruit_dir_path, "*.jpg"))
a
