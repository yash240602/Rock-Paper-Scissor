import cv2
import numpy as np
import mediapipe as mp
import os
import time
import datetime

# Initialize MediaPipe hands with improved settings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_hand(frame, mp_hands_instance):
    """
    Detect hands in the frame using MediaPipe Hands with improved sensitivity
    
    Args:
        frame: Image frame from webcam
        mp_hands_instance: MediaPipe hands instance
        
    Returns:
        results: Hand detection results from MediaPipe
    """
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable 
    rgb_frame.flags.writeable = False
    
    # Process the image and detect hands
    results = mp_hands_instance.process(rgb_frame)
    
    # Mark the image as writeable again
    rgb_frame.flags.writeable = True
    
    # Enhanced error handling for hand detection
    if results is None or not hasattr(results, 'multi_hand_landmarks'):
        # Create empty results object for graceful failure
        return None
    
    return results

def extract_hand_region(frame, results):
    """
    Extract the hand region from the frame using MediaPipe hand landmarks
    
    Args:
        frame: Input camera frame
        results: MediaPipe hand detection results
    
    Returns:
        hand_img: Cropped hand image or None if no hand detected
        success: Boolean indicating if hand was detected
        (x_min, y_min, x_max, y_max): Bounding box coordinates
    """
    if not results or not results.multi_hand_landmarks:
        return None, False, (0, 0, 0, 0)
    
    h, w, _ = frame.shape
    landmarks = results.multi_hand_landmarks[0].landmark
    
    # Calculate bounding box with larger padding
    x_min = w
    y_min = h
    x_max = 0
    y_max = 0
    
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    
    # Add larger padding for more robust detection
    padding = 40  # Increased from 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Extract hand region
    if x_min < x_max and y_min < y_max:
        hand_img = frame[y_min:y_max, x_min:x_max].copy()
        return hand_img, True, (x_min, y_min, x_max, y_max)
    
    return None, False, (0, 0, 0, 0)

def draw_hand_landmarks(frame, results):
    """
    Draw hand landmarks on the frame
    
    Args:
        frame: Image frame from webcam
        results: Hand detection results from MediaPipe
        
    Returns:
        frame: Frame with hand landmarks drawn
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return frame

def extract_hand_features(results):
    """
    Extract hand landmark features from MediaPipe results
    
    Args:
        results: Hand detection results from MediaPipe
        
    Returns:
        features: Normalized landmark coordinates as features
    """
    if not results.multi_hand_landmarks:
        return None
    
    # Get the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Extract the x, y coordinates of each landmark
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return features

def preprocess_landmarks(landmarks):
    """
    Preprocess hand landmarks for classification
    
    Args:
        landmarks: Raw hand landmark features
        
    Returns:
        processed_landmarks: Preprocessed features ready for prediction
    """
    if landmarks is None:
        return None
    
    # Convert to numpy array
    landmarks_array = np.array(landmarks)
    
    # Normalize the landmarks to be between 0 and 1
    min_val = landmarks_array.min()
    max_val = landmarks_array.max()
    
    # Check for division by zero
    if max_val - min_val == 0:
        return landmarks_array.reshape(1, -1)
    
    normalized_landmarks = (landmarks_array - min_val) / (max_val - min_val)
    
    # Reshape for model prediction
    return normalized_landmarks.reshape(1, -1)

def preprocess_image(img, size=(64, 64)):
    """
    Preprocess image for model prediction
    
    Args:
        img: Input image
        size: Target size for resizing
    
    Returns:
        img_gray_flat: Flattened grayscale image
    """
    img_resized = cv2.resize(img, size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray_flat = img_gray.flatten()
    return img_gray_flat

def augment_image(img):
    """
    Apply data augmentation to increase training dataset
    
    Args:
        img: Input image
    
    Returns:
        augmented_images: List of augmented images
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Flip horizontally
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)
    
    # Slight rotation (+/- 5 degrees)
    rows, cols = img.shape[:2]
    for angle in [5, -5]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(rotated)
    
    # Brightness shifts
    bright_shift = cv2.convertScaleAbs(img, alpha=1.05, beta=10)
    augmented_images.append(bright_shift)
    dark_shift = cv2.convertScaleAbs(img, alpha=0.95, beta=-10)
    augmented_images.append(dark_shift)
    
    # Adjust brightness (keep original version as well)
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    augmented_images.append(bright)
    
    # Adjust contrast (keep original version as well)
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    augmented_images.append(contrast)
    
    # Rotation (keep original larger rotations as well)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated_10 = cv2.warpAffine(img, M, (cols, rows))
    augmented_images.append(rotated_10)
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
    rotated_neg10 = cv2.warpAffine(img, M, (cols, rows))
    augmented_images.append(rotated_neg10)
    
    return augmented_images

# UI helper functions
def draw_welcome_screen(frame):
    """
    Draw a welcome screen with instructions
    
    Args:
        frame: Input camera frame
    
    Returns:
        frame: Frame with welcome screen drawn on it
    """
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to draw_welcome_screen")
        return frame
        
    try:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Add semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add title
        cv2.putText(frame, "ROCK PAPER SCISSORS", 
                  (w//2 - 180, h//2 - 80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'p' to play", 
                  (w//2 - 100, h//2 - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit", 
                  (w//2 - 100, h//2 + 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                  
        # Add keyboard fallback instructions
        cv2.putText(frame, "Keyboard Fallback during play:", 
                  (w//2 - 150, h//2 + 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                  
        cv2.putText(frame, "Press '1' for Rock, '2' for Paper, '3' for Scissors", 
                  (w//2 - 230, h//2 + 80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except Exception as e:
        print(f"Error in draw_welcome_screen: {e}")
    
    return frame

def draw_scoreboard(frame, user_score, computer_score):
    """
    Draw a scoreboard showing user and computer scores
    
    Args:
        frame: Input camera frame
        user_score: User's current score
        computer_score: Computer's current score
    
    Returns:
        frame: Frame with scoreboard drawn on it
    """
    # Ensure frame is a valid numpy array before drawing
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to draw_scoreboard")
        return frame
    
    # Draw background for scoreboard
    try:
        cv2.rectangle(frame, (10, 10), (350, 40), (50, 50, 50), -1)
        
        # Add scores
        cv2.putText(frame, f"You: {user_score}", 
                  (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Computer: {computer_score}", 
                  (170, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except Exception as e:
        print(f"Error drawing scoreboard: {e}")
    
    return frame

def draw_hand_box(frame, x_min, y_min, x_max, y_max):
    """
    Draw a bounding box around the detected hand
    
    Args:
        frame: Input camera frame
        x_min, y_min, x_max, y_max: Bounding box coordinates
    
    Returns:
        frame: Frame with hand box drawn on it
    """
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to draw_hand_box")
        return frame
        
    if x_min == x_max or y_min == y_max:
        return frame
    
    try:  
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Hand Detected", 
                  (x_min, y_min - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error in draw_hand_box: {e}")
    
    return frame

def start_countdown(frame, countdown):
    """
    Draw the countdown on the frame
    
    Args:
        frame: Input camera frame
        countdown: Current countdown value
    
    Returns:
        frame: Frame with countdown drawn on it
    """
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to start_countdown")
        return frame
    
    h, w = frame.shape[:2]
    
    # Draw countdown
    cv2.putText(frame, str(max(1, countdown)), 
               (w//2 - 50, h//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 10)
    
    return frame

def display_result(frame, user_move, computer_move, result):
    """
    Display the game result on the frame
    
    Args:
        frame: Input camera frame
        user_move: User's move (rock, paper, scissors)
        computer_move: Computer's move (rock, paper, scissors)
        result: Game result string
    
    Returns:
        frame: Frame with result drawn on it
    """
    if frame is None or not isinstance(frame, np.ndarray):
        print("Invalid frame provided to display_result")
        return frame
    
    try:
        h, w = frame.shape[:2]
        
        # Display user and computer moves
        cv2.putText(frame, f"Your move: {user_move}", 
                  (10, h - 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Computer's move: {computer_move}", 
                  (10, h - 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the result with appropriate color
        if result == "You win!":
            color = (0, 255, 0)
        elif result == "Computer wins!":
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)
        
        cv2.putText(frame, result, 
                  (10, h - 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except Exception as e:
        print(f"Error in display_result: {e}")
    
    return frame

def save_screenshot(frame, folder="screenshots"):
    """
    Save a screenshot to the screenshots folder
    
    Args:
        frame: Input camera frame
        folder: Folder to save screenshot
    
    Returns:
        filename: Name of the saved file
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{folder}/screenshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved to {filename}")
    return filename

def determine_winner(user_move, computer_move):
    """
    Determine the winner of a rock-paper-scissors round
    
    Args:
        user_move: User's move (rock, paper, scissors)
        computer_move: Computer's move (rock, paper, scissors)
    
    Returns:
        result: String indicating the result
    """
    if user_move == computer_move:
        return "It's a tie!"
    elif (user_move == 'rock' and computer_move == 'scissors') or \
         (user_move == 'paper' and computer_move == 'rock') or \
         (user_move == 'scissors' and computer_move == 'paper'):
        return "You win!"
    else:
        return "Computer wins!"

def log_misclassification(frame, actual_gesture, predicted_gesture, confidence, landmarks=None):
    """
    Log and save screenshots of misclassified or low-confidence gestures
    
    Args:
        frame: Image frame showing the hand gesture
        actual_gesture: The actual/ground truth gesture (if known)
        predicted_gesture: The model's predicted gesture
        confidence: Confidence score of the prediction
        landmarks: Hand landmarks (optional, for visualization)
        
    Returns:
        log_path: Path to the saved image for debugging
    """
    # Create directories if they don't exist
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get current timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create informative filename based on prediction results
    if actual_gesture:
        # For training/validation cases where we know the actual gesture
        filename = f"{actual_gesture}_predicted_{predicted_gesture}_{confidence:.2f}_{timestamp}.jpg"
    else:
        # For runtime cases where we only have the prediction
        filename = f"predicted_{predicted_gesture}_{confidence:.2f}_{timestamp}.jpg"
    
    # Prepare the debug frame with annotations
    debug_frame = frame.copy()
    
    # Add text showing prediction details
    cv2.putText(debug_frame, f"Predicted: {predicted_gesture}", 
              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(debug_frame, f"Confidence: {confidence:.2f}", 
              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if actual_gesture:
        cv2.putText(debug_frame, f"Actual: {actual_gesture}", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save the debug image
    log_path = os.path.join(debug_dir, filename)
    cv2.imwrite(log_path, debug_frame)
    
    # Print log message
    print(f"[DEBUG] Saved {log_path} - Predicted: {predicted_gesture}, Confidence: {confidence:.2f}")
    
    return log_path

def save_training_image(frame, gesture, index):
    """
    Save an image for training purposes
    
    Args:
        frame: Image frame from webcam
        gesture: The gesture class ('rock', 'paper', or 'scissors')
        index: Index number for the file
        
    Returns:
        filepath: Path to the saved image
    """
    # Create directory if it doesn't exist
    data_dir = os.path.join("data", gesture)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create filename
    filename = f"{gesture}_{index:04d}.jpg"
    filepath = os.path.join(data_dir, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    
    return filepath

def evaluate_prediction_confidence(prediction_history, threshold=0.7):
    """
    Evaluate the confidence of predictions based on historical predictions
    
    Args:
        prediction_history: List of recent predictions
        threshold: Confidence threshold
        
    Returns:
        prediction: Most confident prediction
        confidence: Confidence score
        is_confident: Boolean indicating if confidence exceeds threshold
    """
    if not prediction_history:
        return None, 0.0, False
    
    # Count occurrences of each prediction
    prediction_counts = {}
    for pred in prediction_history:
        if pred in prediction_counts:
            prediction_counts[pred] += 1
        else:
            prediction_counts[pred] = 1
    
    # Get the most common prediction
    prediction = max(prediction_counts, key=prediction_counts.get)
    
    # Calculate confidence
    confidence = prediction_counts[prediction] / len(prediction_history)
    
    return prediction, confidence, confidence >= threshold 