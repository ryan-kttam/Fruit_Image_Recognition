import numpy as np
import cv2
import glob
import os
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential, Model
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.ndimage import rotate as rot

ndim = 100

def process_image(img_path, img_list, label_list, fruit_name, dim):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (dim, dim))
    img_list.append((image / 255.).tolist())

    image_flip_vert = np.flipud(image)
    img_list.append((image_flip_vert / 255.).tolist())

    image_rotate90 = rot(image, 90)
    img_list.append((image_rotate90 / 255.).tolist())

    image_rotate270 = rot(image, 270)
    img_list.append((image_rotate270 / 255.).tolist())

    label_list += [fruit_name]*4

train_img = []
train_label = []
valid_img = []
valid_label = []
test_img = []
test_label = []

def reading_images(img_folder, dim):
    # reading_images: a function that does the following steps:
    # 1. store the image pixels to fruit_img, and the label names to fruit_labels
    # 2. Read every image
    # 3. resize the images
    # 4. standardize the images
    # 5. return a list of image pixels and a list of labels
    for fruit_dir_path in glob.glob(img_folder):
        fruit_name = fruit_dir_path.split("\\")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            num = np.random.choice(np.arange(0, 3), p=[0.8, 0.15, 0.05])
            if num == 0:
                process_image(image_path, train_img, train_label, fruit_name, dim)
            elif num == 1:
                process_image(image_path, valid_img, valid_label, fruit_name, dim)
            else:
                process_image(image_path, test_img, test_label, fruit_name, dim)

        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpeg")):
            num = np.random.choice(np.arange(0, 3), p=[0.8, 0.15, 0.05])
            if num == 0:
                process_image(image_path, train_img, train_label, fruit_name, dim)
            elif num == 1:
                process_image(image_path, valid_img, valid_label, fruit_name, dim)
            else:
                process_image(image_path, test_img, test_label, fruit_name, dim)

        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
            num = np.random.choice(np.arange(0, 3), p=[0.8, 0.15, 0.05])
            if num == 0:
                process_image(image_path, train_img, train_label, fruit_name, dim)
            elif num == 1:
                process_image(image_path, valid_img, valid_label, fruit_name, dim)
            else:
                process_image(image_path, test_img, test_label, fruit_name, dim)

reading_images(".../fruits/*", ndim)

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

encoded_train_labels = np.array(helper_label(train_label))
encoded_valid_labels = np.array(helper_label(valid_label))
encoded_test_labels = np.array(helper_label(test_label))



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

    check = ModelCheckpoint("C:/Users/Ryan/Desktop/Python_Projects/Fruit recognition v2/checkpt.h5",
                            monitor='val_acc', save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)

    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, mode='auto')

    cnn_model.fit(train_img, encoded_train_labels,
                  validation_data=(valid_img, encoded_valid_labels),
                  batch_size=100, epochs=50, callbacks=[check, early_stop])

    cnn_model.load_weights("C:/Users/Ryan/Desktop/Python_Projects/Fruit recognition v2/checkpt.h5")

    if model_type == "scratch":
        valid_pred_class = cnn_model.predict_classes(valid_img)
        scratch_number_correct = 0
        for i in range(len(valid_pred_class)):
            if valid_pred_class[i] == np.where(encoded_valid_labels == 1)[1][i]:
                scratch_number_correct += 1
        print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate,
                                                                                         scratch_number_correct / len(
                                                                                             valid_pred_class) * 100))

    elif model_type == "pre_trained":
        pre_trained_prediction = cnn_model.predict(valid_img)
        pre_trained_number_correct = 0
        for i in range(len(pre_trained_prediction)):
            if np.where(pre_trained_prediction[i] == max(pre_trained_prediction[i])) == np.where(encoded_valid_labels[i] == 1):
                pre_trained_number_correct += 1
        print("For {} with learning rate {}, The accuracy on test set is {:.2f}%".format(optimizer, learn_rate,
                                                                                         pre_trained_number_correct / len(
                                                                                             pre_trained_prediction) * 100))
    return cnn_model

# -------------------------- T R A N S F E R   L E A R N I N G ----------------------------

def pre_trained(trainable, dim):  # trainable is a boolean
    # create a pre-trained model (VGG16)
    # trainable: TRUE/ FALSE
    #   TRUE means the last five layers are trainable
    #   FALSE means non of the layers are trainable
    model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(dim, dim, 3))
    if not trainable:
        for layer in model.layers[:]:
            layer.trainable = False
    else:
        for layer in model.layers[:-5]:
            layer.trainable = False
    x = model.output
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(22, activation='softmax')(x)
    pre_trained_model = Model(inputs=model.input, output=x)
    pre_trained_model.summary()
    return pre_trained_model


optimizer = 'rms'
learn_rate = 0.001

transfered_model = fine_tuning(pre_trained(True, ndim), "sgd", 0.01, "pre_trained")


# None, 0.001, RMS: 72.56%
# None, 0.01, RMS: 3.72%
# None, 0.0001, RMS: 72.35%
# last 5, 0.001, RMS: 6.88%
# last 5, 0.01, RMS: 3.72%
# last 5, 0.0001, RMS: 81.73%
# last 5, 0.00001, RMS: 78.72%
# None, 0.001, SGD: 60.74%
# None, 0.01, SGD: 70.27%
# None, 0.0001, SGD: 33.46%
# last 5, 0.001, RMS: 76.65%
# last 5, 0.01, RMS: 79.51%
# last 5, 0.0001, RMS: 56.59%

len(encoded_test_labels)
# Apply the best model to test set
# test_result: real label, number of position where 1 occurs (from 0-40)
# np.where(labels_test == 1): pred label, number of position where 1 occurs (from 0-40)
test_result = transfered_model.predict(test_img)
number_correct = 0
for i in range(len(test_result)):
    if np.where(test_result[i] == max(test_result[i])) == np.where(encoded_test_labels== 1)[1][i]:
        number_correct += 1
print("The accuracy on test set is {:.2f}%".format(number_correct / len(test_result) * 100))

# Apply the model model to self-taken images


def process_image_real_life(img_folder, dim):
    for fruit_dir_path in glob.glob(img_folder):
        print (fruit_dir_path )
        fruit_name = fruit_dir_path.split("\\")[-1]
        print (fruit_name)
        image = cv2.imread(fruit_dir_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (dim, dim))
        real_life_img.append((image / 255.).tolist())
        real_life_labels.append(fruit_name)

real_life_img = []
real_life_labels = []

process_image_real_life("C:/real_life_test/test/*", ndim)

real_life_img, real_life_labels = reading_images(real_life_img, real_life_labels,
                                                 "C:/real_life_test/test/*")

t_model = pre_trained(False, 100)
t_model.load_weights("C:/Fruit recognition v2/checkpt.h5")
prediction = t_model.predict(real_life_img)
number_correct = 0
np.where(prediction[0] == max(prediction[0]))

#[int(i) for i in np.where(prediction[0] == max(prediction[0]))]

for i in np.where(prediction[0] == max(prediction[0])):
    idx = int(i)

sorted(set(train_label))[idx]

for i in range(len(prediction)):
    for j in np.where(prediction[i] == max(prediction[i])):
        idx = int(j)
    print ("Predicting ", real_life_labels[i], " as ", sorted(set(train_label))[idx])




# -------------------------trying different pre-trained model

applications.Xception()
model = applications.Xception(weights="imagenet", include_top=False, input_shape=(139, 139, 3), pooling='max')
for layer in model.layers[:]:
    layer.trainable = False

x = model.output
#x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(22, activation='softmax')(x)
pre_trained_model = Model(inputs=model.input, output=x)
pre_trained_model.summary()

fine_tuning(pre_trained_model, "rms", 0.001, "pre_trained")

