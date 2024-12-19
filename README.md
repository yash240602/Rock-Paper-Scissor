# Rock-Paper-scissor-Game-
**Rock Paper Scissors Game using OpenCV and scikit-learn**

This project is a simple implementation of the classic Rock Paper Scissors game using Python, OpenCV for image processing, and scikit-learn for machine learning-based gesture recognition. 

### Dependencies:
- Python 3.x
- OpenCV (cv2)
- scikit-learn
- numpy

### Installation:
1. Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. Install the required dependencies using pip:
   ```
   pip install opencv-python scikit-learn numpy
   ```

3. Clone or download this repository to your local machine.

### Usage:
1. Navigate to the directory where you have downloaded or cloned the repository.

2. Run the game script:
   ```
   python rock_paper_scissors.py
   ```

3. Follow the on-screen instructions to play the game. The game will prompt you to show your hand gesture (Rock, Paper, or Scissors) to the webcam, which will then be recognized by the machine learning model.

### How it works:
- The game uses the OpenCV library to capture video from the webcam.
- It preprocesses the video frames to extract the region of interest (ROI) containing the hand gesture.
- Feature extraction techniques are applied to convert the ROI into a format suitable for machine learning.
- The preprocessed data is then fed into a scikit-learn machine learning model trained to recognize Rock, Paper, and Scissors gestures.
- Finally, the recognized gesture is compared against the user's input, and the winner of the round is determined.

### Dataset:
- The dataset used to train the machine learning model consists of images of hand gestures for Rock, Paper, and Scissors.
- These images were collected and labeled manually or using automated scripts.
- The dataset was split into training and testing sets to train and evaluate the model's performance.

### Acknowledgments:
- This project was inspired by the interest in combining computer vision and machine learning to create interactive applications.
- Special thanks to the developers of OpenCV and scikit-learn for providing powerful tools for image processing and machine learning.

### Disclaimer:
This project is intended for educational purposes only. The accuracy of the gesture recognition system may vary depending on factors such as lighting conditions, background clutter, and the user's hand size and position.
