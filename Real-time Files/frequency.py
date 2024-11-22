import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import time

global loadedModel
size = 300

def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    currentFrame = currentFrame.reshape(1, size, size, 1).astype('float32')
    currentFrame = currentFrame / 255.0
    prob = loadedModel.predict(currentFrame)
    max_prob = np.max(prob[0])
    if max_prob > 0.90:
        predicted_class = np.argmax(prob[0])
        return predicted_class, max_prob
    return "none", 0

if __name__ == '__main__':
    loadedModel = load_model(
        "D:\\Pothole_detection\\pothole-detection-system-using-convolution-neural-networks\\Real-time Files\\latest_full_model.h5")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Camera not accessible.")
        exit()

    show_pred = False
    pothole_count = 0
    start_time = time.time()
    last_detection_time = 0
    detection_cooldown = 1  # 1 second cooldown between detections

    while True:
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()
        grayClone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

        pothole, prob = predict_pothole(grayClone)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if pothole == 1 and current_time - last_detection_time > detection_cooldown:
            pothole_count += 1
            last_detection_time = current_time

        keypress_toshow = cv2.waitKey(1)

        if keypress_toshow == ord("e"):
            show_pred = not show_pred

        if show_pred:
            cv2.putText(clone, f"{pothole} {prob * 100:.2f}%", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        # Display pothole frequency
        cv2.putText(clone, f"Potholes: {pothole_count} in {elapsed_time:.1f}s", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)

        cv2.imshow("GrayClone", grayClone)
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(10) & 0xFF

        if keypress == ord("q"):
            break

        # Reset count every 30 seconds
        if elapsed_time >= 30:
            print(f"Detected {pothole_count} potholes in 30 seconds")
            pothole_count = 0
            start_time = current_time

    camera.release()
    cv2.destroyAllWindows()