import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

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
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img.flatten())
        labels.append(label)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the trained model
import pickle
with open('rps_model.pkl', 'wb') as f:
    pickle.dump(knn, f)