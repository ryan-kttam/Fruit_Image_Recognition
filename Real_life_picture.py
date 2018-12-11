import numpy as np
import cv2
import glob
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model
from keras import applications


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


def process_image_real_life(img_folder, dim):
    for fruit_dir_path in glob.glob(img_folder):
        print (fruit_dir_path )
        fruit_name = fruit_dir_path.split("\\")[-1]
        print (fruit_name)
        image = cv2.imread(fruit_dir_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (dim, dim))
        real_life_img.append((image / 255.).tolist())
        real_life_labels.append(fruit_name)


def read_label():
    fruit_list = []
    for fruit_dir_path in glob.glob('C:/Users/Ryan/Desktop/Udacity Machine Learning/capstone/new_fruit/fruits/*'):
        fruit_list.append(fruit_dir_path.split("\\")[-1])
    return fruit_list


real_life_img = []
real_life_labels = []

process_image_real_life("C:/Users/Ryan/Desktop/Python_Projects/Fruit recognition v2/test/*", 100)


fruit_label = read_label()

t_model = pre_trained(True, 100)
t_model.load_weights("C:/Users/Ryan/Desktop/Python_Projects/Fruit recognition v2/checkpt.h5")
prediction = t_model.predict(real_life_img)


for i in range(len(prediction)):
    for j in np.where(prediction[i] == max(prediction[i])):
        idx = int(j)
    print("Predicting ", real_life_labels[i], " as ", fruit_list[idx])
