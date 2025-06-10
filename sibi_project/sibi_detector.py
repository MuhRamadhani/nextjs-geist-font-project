import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('sibi_model.h5')  # Replace with your model path

# Load the class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def detect_hand_and_recognize_sign(image):
    """
    Detects the hand in the image and recognizes the sign.
    """
    # Hand detection using OpenCV (Haar cascade classifier)
    hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')  # Replace with your Haar cascade path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in hands:
        # Extract the hand region
        hand_roi = image[y:y+h, x:x+w]

        # Preprocess the hand region
        resized_hand = cv2.resize(hand_roi, (100, 100))
        expanded_hand = np.expand_dims(resized_hand, axis=0)
        normalized_hand = expanded_hand / 255.0

        # Predict the sign
        prediction = model.predict(normalized_hand)
        predicted_class = np.argmax(prediction)
        sign = class_names[predicted_class]

        # Draw a rectangle around the hand and display the sign
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, sign, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Detect hand and recognize sign
        processed_frame = detect_hand_and_recognize_sign(frame)

        # Display the processed frame
        cv2.imshow('SIBI Detector', processed_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
