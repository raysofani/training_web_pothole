import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

global loadedModel
size = 300


# Resize the frame to required dimensions and predict
def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    currentFrame = currentFrame.reshape(1, size, size, 1).astype('float32')
    currentFrame = currentFrame / 255.0
    prob = loadedModel.predict(currentFrame)
    max_prob = np.max(prob[0])  # Get the maximum probability
    if max_prob > 0.90:
        predicted_class = np.argmax(prob[0])  # Get the predicted class
        return predicted_class, max_prob
    return "none", 0


# Main function
if __name__ == '__main__':
    # Load the trained model
    loadedModel = load_model(
        "D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\Real-time Files\\latest_full_model.h5")

    # Initialize video capture
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Camera not accessible.")
        exit()

    show_pred = False

    # Loop until interrupted
    while True:
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()
        (height, width) = frame.shape[:2]

        grayClone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

        # Predict pothole
        pothole, prob = predict_pothole(grayClone)

        keypress_toshow = cv2.waitKey(1)

        if keypress_toshow == ord("e"):
            show_pred = not show_pred
            print(f"Show Predictions: {show_pred}")  # Debug print

        if show_pred:
            cv2.putText(clone, f"{pothole} {prob * 100:.2f}%", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("GrayClone", grayClone)
        cv2.imshow("Video Feed", clone)

        # Check for key presses
        keypress = cv2.waitKey(10) & 0xFF  # Increase delay for responsiveness

        print(f"Key Pressed: {keypress}")  # Debug print
        if keypress == ord("q"):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
