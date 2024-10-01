import numpy as np
import cv2
import glob
from keras.models import load_model
from keras.utils import to_categorical

# Global variable for image size
global size
size = 100

# Load the trained model
model = load_model("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/sample.h5")

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Plain/*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
for i in range(len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("D:/Pothole_detection/pothole-detection-system-using-convolution-neural-networks/My Dataset/test/Pothole/*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
for i in range(len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

# Combine test data
X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)

# Reshape test data
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Prepare labels for the test data
y_test1 = np.ones([temp3.shape[0]], dtype=int)  # Pothole labels
y_test2 = np.zeros([temp4.shape[0]], dtype=int)  # Non-pothole labels
y_test = np.concatenate((y_test1, y_test2))
y_test = to_categorical(y_test)  # Convert to categorical format

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)  # Get class indices

# Output results
for i in range(len(X_test)):
    print(f">>> Image {i+1}: Predicted Class = {predicted_classes[i]}")

# Optional: Evaluate the model (uncomment if needed)
# metrics = model.evaluate(X_test, y_test)
# for metric_i in range(len(model.metrics_names)):
#     metric_name = model.metrics_names[metric_i]
#     metric_value = metrics[metric_i]
#     print(f'{metric_name}: {metric_value}')
