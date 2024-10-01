import numpy as np
import cv2
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Global size
size = 300

# Load the trained model
model_path = "D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\Real-time Files\\latest_full_model.h5"
model = load_model(model_path)

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\test\\Plain\\*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
for i in range(len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\My Dataset\\test\\Pothole\\*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
for i in range(len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

# Combine the test sets
X_test = np.concatenate((temp3, temp4), axis=0)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Prepare labels
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)
y_test = np.concatenate((y_test1, y_test2), axis=0)
y_test = to_categorical(y_test)

# Normalize test data
X_test = X_test / 255.0

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Print results
for i in range(len(X_test)):
    print(f">>> Predicted {i} = {predicted_classes[i]}")

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print("Test Accuracy:", metrics[1] * 100, "%")
