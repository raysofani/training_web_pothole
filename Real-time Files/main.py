import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# Global variables
global inputShape, size

def kerasModel4():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size, size, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

size = 300

# Load Training data: pothole
potholeTrainImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\train\\Pothole\\*.jpg")
potholeTrainImages.extend(glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\train\\Pothole\\*.jpeg"))
potholeTrainImages.extend(glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\train\\Pothole\\*.png"))

train1 = [cv2.imread(img, 0) for img in potholeTrainImages]
for i in range(0, len(train1)):
    train1[i] = cv2.resize(train1[i], (size, size))
temp1 = np.asarray(train1)

# Load Training data: non-pothole
nonPotholeTrainImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\train\\Plain\\*.jpg")
train2 = [cv2.imread(img, 0) for img in nonPotholeTrainImages]
for i in range(0, len(train2)):
    train2[i] = cv2.resize(train2[i], (size, size))
temp2 = np.asarray(train2)

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\test\\Plain\\*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
for i in range(0, len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\test\\Pothole\\*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
for i in range(0, len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

# Prepare training and testing datasets
X_train = np.asarray(train1 + train2)
X_test = np.asarray(test1 + test2)

y_train1 = np.ones([temp1.shape[0]], dtype=int)
y_train2 = np.zeros([temp2.shape[0]], dtype=int)
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)

y_train = np.asarray(np.concatenate((y_train1, y_train2)))
y_test = np.asarray(np.concatenate((y_test1, y_test2)))

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 1)
model = kerasModel4()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1)

# Evaluate the model
metricsTrain = model.evaluate(X_train, y_train)
print("Training Accuracy: ", metricsTrain[1] * 100, "%")

metricsTest = model.evaluate(X_test, y_test)
print("Testing Accuracy: ", metricsTest[1] * 100, "%")

# Save the model
print("Saving model weights and configuration file")
model.save('latest_full_model.h5')
print("Saved model to disk")
