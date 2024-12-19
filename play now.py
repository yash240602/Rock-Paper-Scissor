import cv2
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pickle
import random
import os

def get_computer_move():
    moves = ['rock', 'paper', 'scissors']
    return random.choice(moves)

# Set the paths for the training data folders
train_data_dir = "C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\rock","C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Paper","C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Scissor"
rock_dir =  (r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\rock")
paper_dir = (r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Paper")
scissors_dir =(r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Scissor")

# Load the images and labels
images = []
labels = []

for folder, label in [(rock_dir, 'rock'), (paper_dir, 'paper'), (scissors_dir, 'scissors')]:
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale images
        img = cv2.resize(img, (64, 64))  # Resize images to a consistent size
        images.append(img.flatten())
        labels.append(label)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Apply dimensionality reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the trained model and PCA object
import pickle
with open('rps_model.pkl', 'wb') as f:
    pickle.dump((knn, pca), f)

# ... (your existing code for loading images and training the model) ...

# Save the trained model and PCA object
with open('rps_model.pkl', 'wb') as f:
    pickle.dump((knn, pca), f)
    import cv2
import numpy as np
import pickle

# Load the trained model and PCA object
with open('rps_model.pkl', 'rb') as f:
    knn, pca = pickle.load(f)

# ... (the rest of your code for prediction and playing the game) ...

# Load the trained model and PCA object
with open('rps_model.pkl', 'rb') as f:
    knn, pca = pickle.load(f)

# Define the class labels
class_labels = ['rock', 'paper', 'scissors']

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    img = pca.transform([img])  # Apply PCA transformation

    # Make a prediction
    prediction = knn.predict(img)
    user_move = class_labels[prediction[0]]

    # Get the computer's move
    computer_move = get_computer_move()

    # Determine the winner
    if user_move == computer_move:
        result = "It's a tie!"
    elif (user_move == 'rock' and computer_move == 'scissors') or \
         (user_move == 'paper' and computer_move == 'rock') or \
         (user_move == 'scissors' and computer_move == 'paper'):
        result = "You win!"
    else:
        result = "Computer wins!"

    # Display the moves and the result
    cv2.putText(frame, f"Your move: {user_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Computer's move: {computer_move}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, result, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Rock Paper Scissors', frame)

    time.sleep(8)

    # Press 'q' to exit
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()