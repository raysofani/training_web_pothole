import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time

import tensorflow as tf
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Cropping2D, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

global inputShape, size

def kerasModel4():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size, size, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(.1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

size = 100

# Load Training data: pothole
potholeTrainImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/train/Pothole/*.jpg")
potholeTrainImages.extend(glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/train/Pothole/*.jpeg"))
potholeTrainImages.extend(glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/train/Pothole/*.png"))

train1 = [cv2.imread(img, 0) for img in potholeTrainImages]
train1 = [cv2.resize(img, (size, size)) for img in train1]
temp1 = np.asarray(train1)

# Load Training data: non-pothole
nonPotholeTrainImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/train/Plain/*.jpg")
train2 = [cv2.imread(img, 0) for img in nonPotholeTrainImages]
train2 = [cv2.resize(img, (size, size)) for img in train2]
temp2 = np.asarray(train2)

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Plain/*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
test2 = [cv2.resize(img, (size, size)) for img in test2]
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Pothole/*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
test1 = [cv2.resize(img, (size, size)) for img in test1]
temp3 = np.asarray(test1)

# Prepare training and testing datasets
X_train = np.concatenate((temp1, temp2), axis=0)
X_test = np.concatenate((temp3, temp4), axis=0)

y_train1 = np.ones(temp1.shape[0], dtype=int)  # Pothole labels
y_train2 = np.zeros(temp2.shape[0], dtype=int)  # Non-pothole labels
y_test1 = np.ones(temp3.shape[0], dtype=int)    # Pothole test labels
y_test2 = np.zeros(temp4.shape[0], dtype=int)   # Non-pothole test labels

# Combine training and testing labels
y_train = np.concatenate((y_train1, y_train2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)

# Shuffle the datasets
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Reshape the datasets for the model
X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Convert labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Training shape X:", X_train.shape)
print("Training shape y:", y_train.shape)

# Build and compile the model
model = kerasModel4()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=500, validation_split=0.1)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

# Save model weights and configuration file
model.save('sample.h5')  # Save the model architecture and weights
model_json = model.to_json()
with open("truesample.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights with the correct extension
model.save_weights("truesample.weights.h5")  # Corrected filename
print("Saved model to disk")
