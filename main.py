import cv2
import numpy as np
import pickle

# Load the trained model
with open('rps_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# Define the class labels
class_labels = ['rock', 'paper', 'scissors']

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()

    # Make a prediction
    prediction = knn.predict([img])
    predicted_class = class_labels[prediction[0]]

    # Display the prediction
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Rock Paper Scissors', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()