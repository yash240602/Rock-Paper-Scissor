import cv2
import os
import time

# Set the paths for the training data folders
train_data_dir = "C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\rock","C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Paper","C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Scissor"
rock_dir =  (r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\rock")
paper_dir = (r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Paper")
scissors_dir =(r"C:\\Users\\aarya\\OneDrive\\Desktop\\Game\\Scissor")

# Create the folders if they don't exist
for folder in [rock_dir, paper_dir, scissors_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Capture images for rock
print("Capturing images for 'Rock'")
for i in range(25):
    ret, frame = cap.read()
    cv2.putText(frame, f"Show 'Rock' gesture (Image {i+1}/25)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Capturing', frame)
    
    # Wait for 5 seconds or until 'c' is pressed
    start_time = time.time()
    while True:
        if cv2.waitKey(1) == ord('c') or time.time() - start_time > 5:
            break
    
    img_path = os.path.join(rock_dir, f'rock_{len(os.listdir(rock_dir))}.jpg')
    cv2.imwrite(img_path, frame)
    print(f'Image saved to {img_path}')

# Capture images for paper
print("Capturing images for 'Paper'")
for i in range(25):
    ret, frame = cap.read()
    cv2.putText(frame, f"Show 'Paper' gesture (Image {i+1}/25)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Capturing', frame)
    
    # Wait for 5 seconds or until 'c' is pressed
    start_time = time.time()
    while True:
        if cv2.waitKey(1) == ord('c') or time.time() - start_time > 5:
            break
    
    img_path = os.path.join(paper_dir, f'paper_{len(os.listdir(paper_dir))}.jpg')
    cv2.imwrite(img_path, frame)
    print(f'Image saved to {img_path}')

# Capture images for scissors
print("Capturing images for 'Scissors'")
for i in range(25):
    ret, frame = cap.read()
    cv2.putText(frame, f"Show 'Scissors' gesture (Image {i+1}/25)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Capturing', frame)
    
    # Wait for 5 seconds or until 'c' is pressed
    start_time = time.time()
    while True:
        if cv2.waitKey(1) == ord('c') or time.time() - start_time > 5:
            break
    
    img_path = os.path.join(scissors_dir, f'scissors_{len(os.listdir(scissors_dir))}.jpg')
    cv2.imwrite(img_path, frame)
    print(f'Image saved to {img_path}')

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()