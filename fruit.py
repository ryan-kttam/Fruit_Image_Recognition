import numpy as np
import cv2
import glob
import os
import random
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential, Model
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
import collections
from matplotlib import pyplot


def reading_images(fruit_img, fruit_labels, img_folder):
    # reading_images: a function that does the following steps:
    # 1. store the image pixels to fruit_img, and the label names to fruit_labels
    # 2. Read every image
    # 3. resize the images
    # 4. standardize the images
    # 5. return a list of image pixels and a list of labels
    for fruit_dir_path in glob.glob(img_folder):
        fruit_name = fruit_dir_path.split("\\")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpeg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (100, 100))
            fruit_img.append((image / 255.).tolist())
            fruit_labels.append(fruit_name)
    return fruit_img, fruit_labels


# creating an empty list for storing the images and labels
fruit_images = []
labels = []

fruit_images, labels = reading_images(fruit_images, labels,
                                      "C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*")


def helper_label(list_of_labels):
    # helper_label: A helper function that help me encode the labels to 0,1
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    encoder = LabelEncoder()
    # turn the labels to 0 - 21
    encoder_int = encoder.fit_transform(list_of_labels)
    encoder_int = encoder_int.reshape(len(encoder_int), 1)
    one_hot_encoder = OneHotEncoder(sparse=False)
    labels_encoded = one_hot_encoder.fit_transform(encoder_int)
    return labels_encoded


encoded_labels = helper_label(labels)


# bar chart by label frequency
fruit_counts = collections.Counter(labels)
pyplot.bar(range(len(fruit_counts)), list(fruit_counts.values()) )
pyplot.xticks(range(len(fruit_counts)), list(fruit_counts.keys()) , rotation=90)
pyplot.title("Fruit Frequency in counts")

# generate random index numbers
# randomize the index
idx = random.sample(range(len(labels)), len(labels))
# Turn into array for later modification
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

# -------------------------- B E N C H M A R K   M O D E L  ------------------------
# constructing a benchmark model
# 1. flatten the pixels
# 2. add two hidden layers with 64 nodes, and ReLU as activation
# 3. after each hidden layer, add a dropout layet with 0.2 probability
# 4. add a final softmax layer with 22 nodes
benchmark_model = Sequential()
benchmark_model.add(Flatten(input_shape=image_train.shape[1:]))  # image_train.shape[1:] is (100,100,3)
benchmark_model.add(Dense(64, activation='relu'))
benchmark_model.add(Dropout(0.2))
benchmark_model.add(Dense(64, activation='relu'))
benchmark_model.add(Dropout(0.2))
benchmark_model.add(Dense(22, activation='softmax'))
benchmark_model.summary()

benchmark_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
Bench_check = ModelCheckpoint("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5", monitor='val_acc',
                              save_best_only=True, save_weights_only=False, mode='auto', period=1)
Bench_early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, mode='auto')
benchmark_model.fit(image_train, labels_train,
                    validation_data=(image_valid, labels_valid),
                    batch_size=50, epochs=20, callbacks=[Bench_check, Bench_early_stop])
benchmark_model.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")
prediction = benchmark_model.predict(image_valid)
number_correct = 0
for i in range(len(prediction)):
    if np.where(prediction[i] == max(prediction[i])) == np.where(labels_valid[i] == 1):
        number_correct += 1
# the accuracy on the validation set is 5.97%
print("For Benchmark model, The accuracy on validation set is {:.2f}%".format(number_correct / len(prediction) * 100))

# ------------------ H E L P E R   F U N C T I O N S  ----------------


def optimizer_generator(optimizer, learn_rate=0.01):
    # optimizer_generator is a helper function that takes the name of the optimizer (rms or sgd) and learning rate as input,
    # then it generates an optimizer
    # optimizer: "rms" or "sgd"
    # learn_rate: default to 0.01
    if optimizer not in ("rms", "sgd"):
        return "please enter rms or sgd as optimizer"
    if optimizer == "rms":
        return optimizers.RMSprop(lr=learn_rate)  # or 0.001 etc
    else:
        return optimizers.SGD(lr=learn_rate)


def fine_tuning(cnn_model, optimizer, learn_rate, model_type=""):
    # fine_tuning is a function that compile a provided CNN model using a provided optimizer name and learning rates.
    # cnn_model: accept any type of cnn model
    # optimizer: "rms" or "sgd"
    # learn_rate: 0.1, 0.01, or 0.001
    # model_type: "scratch" or "pre_trained"

    if model_type not in ("scratch", "pre_trained"):
        return "please specify which model you are training. (scratch or pre_trained)"

    model_optimizer = optimizer_generator(optimizer, learn_rate)
    cnn_model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    check = ModelCheckpoint("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5",
                            monitor='val_acc', save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, mode='auto')

    cnn_model.fit(image_train, labels_train,
                  validation_data=(image_valid, labels_valid),
                  batch_size=100, epochs=50, callbacks=[check, early_stop])
    cnn_model.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")

    if model_type == "scratch":
        valid_pred_class = cnn_model.predict_classes(image_valid)
        scratch_number_correct = 0
        for i in range(len(labels_valid)):
            if valid_pred_class[i] == np.where(labels_valid == 1)[1][i]:
                scratch_number_correct += 1
        print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate,
                                                                                         scratch_number_correct / len(
                                                                                             valid_pred_class) * 100))

    elif model_type == "pre_trained":
        pre_trained_prediction = cnn_model.predict(image_valid)
        pre_trained_number_correct = 0
        for i in range(len(pre_trained_prediction)):
            if np.where(pre_trained_prediction[i] == max(pre_trained_prediction[i])) == np.where(labels_valid[i] == 1):
                pre_trained_number_correct += 1
        print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate,
                                                                                         pre_trained_number_correct / len(
                                                                                             pre_trained_prediction) * 100))
    return cnn_model


# ----------------M O D E L   F R O M   S C R A T C H ---------------------------
# train the model

def cnn_from_scratch():
    # create a CNN model from scratch
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


m = fine_tuning(cnn_from_scratch(), optimizer="rms",
                learn_rate=0.01, model_type="scratch")

m.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")

# make prediction on validation set to get the validation accuracy
scratch_prediction = m.predict_classes(image_valid)
scratch_prediction_correct = 0
for i in range(len(labels_valid)):
    if scratch_prediction[i] == np.where(labels_valid == 1)[1][i]:
        scratch_prediction_correct += 1


# -------------------------- T R A N S F E R   L E A R N I N G ----------------------------


def pre_trained(trainable):  # trainable is a boolean
    # create a pre-trained model (VGG16)
    # trainable: TRUE/ FALSE
    #   TRUE means the last five layers are trainable
    #   FALSE means non of the layers are trainable
    model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
    if not trainable:
        for layer in model.layers[:]:
            layer.trainable = False
    else:
        for layer in model.layers[:-5]:
            layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(22, activation='softmax')(x)
    pre_trained_model = Model(inputs=model.input, output=x)
    pre_trained_model.summary()
    return pre_trained_model


transfered_model = fine_tuning(pre_trained(False), "rms", 0.001, "pre_trained")

# make prediction on validation set to get the validation accuracy
pred = transfered_model.predict(image_valid)
pred_correct = 0
for i in range(len(pred)):
    if np.where(pred[i] == max(pred[i])) == np.where(labels_valid[i] == 1):
        pred_correct += 1
print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate,
                                                                                 pred_correct / len(
                                                                                     labels_valid) * 100))

# Apply the best model to test set
# test_result: real label, number of position where 1 occurs (from 0-40)
# np.where(labels_test == 1): pred label, number of position where 1 occurs (from 0-40)
test_result = transfered_model.predict_classes(image_test)
number_correct = 0
for i in range(len(test_result)):
    if test_result[i] == np.where(labels_test == 1)[1][i]:
        number_correct += 1
print("The accuracy on test set is {:.2f}%".format(number_correct / len(test_result) * 100))

# Apply the model model to self-taken images
real_life_img = []
real_life_labels = []
real_life_img, real_life_labels = reading_images(real_life_img, real_life_labels,
                                                 "C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/real_life_test/*")
t_model = pre_trained(False)
t_model.load_weights("C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/checkpt.h5")
prediction = t_model.predict(image_test)
number_correct = 0
for i in range(len(prediction)):
    if np.where(prediction[i] == max(prediction[i])) == np.where(labels_test[i] == 1):
        number_correct += 1
    else:  # show which ones are predicted wrong
        plt.imshow(image_test[i])
        print("For position: {}, Prediction: {}, correct answer is: {}".format(i,
                                                                               np.where(
                                                                                   prediction[i] == max(prediction[i])),
                                                                               np.where(labels_test[i] == 1)))
