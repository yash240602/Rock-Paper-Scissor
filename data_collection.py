import cv2
import os
import time
import numpy as np
import mediapipe as mp
from utils import (
    detect_hand, 
    draw_hand_landmarks, 
    extract_hand_region, 
    draw_hand_box,
    save_screenshot
)

def create_directories():
    """Create directories for storing training images"""
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the paths for the training data folders
    data_dir = os.path.join(base_dir, "data")
    rock_dir = os.path.join(data_dir, "rock")
    paper_dir = os.path.join(data_dir, "paper")
    scissors_dir = os.path.join(data_dir, "scissors")
    
    # Create the folders if they don't exist
    for folder in [data_dir, rock_dir, paper_dir, scissors_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
            
    return rock_dir, paper_dir, scissors_dir

def save_image_with_hand(frame, save_dir, prefix, index, results):
    """Save an image with detected hand"""
    if not results.multi_hand_landmarks:
        return False, "No hand detected"
    
    # Extract hand region
    hand_img, success, bbox = extract_hand_region(frame, results)
    if not success:
        return False, "Failed to extract hand region"
    
    # Generate filename with index
    img_path = os.path.join(save_dir, f"{prefix}_{index}.jpg")
    
    # Save both the full frame and the cropped hand
    cv2.imwrite(img_path, frame)
    
    # Also save the cropped hand image with "_cropped" suffix
    cropped_path = os.path.join(save_dir, f"{prefix}_{index}_cropped.jpg")
    cv2.imwrite(cropped_path, hand_img)
    
    print(f"Images saved to {img_path} and {cropped_path}")
    return True, img_path

def capture_training_images(debug=False):
    """Capture and save training images for rock, paper, scissors gestures"""
    # Create directories
    rock_dir, paper_dir, scissors_dir = create_directories()
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\n=== DATA COLLECTION TIPS ===")
    print("Tip: Vary angles and lighting while collecting data.")
    print("Move your hand closer/farther and tilt it for better generalization.")
    print("Try different backgrounds to make your model more robust.")
    print("===============================\n")
    
    # Set up MediaPipe
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        gestures = [
            ("rock", rock_dir),
            ("paper", paper_dir),
            ("scissors", scissors_dir)
        ]
        
        for gesture_name, gesture_dir in gestures:
            num_images = 25  # Number of images to capture per gesture
            count = 0
            
            print(f"\n=== Capturing images for '{gesture_name.upper()}' gesture ===")
            print(f"Show your {gesture_name} gesture to the camera.")
            print("Press 'c' to capture or wait for the timer.")
            print("Press 'n' to move to the next gesture.")
            print("Press 'q' to quit.\n")
            
            waiting_time = 3  # seconds to wait before auto-capture
            
            while count < num_images:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Detect hand
                results = detect_hand(frame, hands)
                
                # Draw hand landmarks
                display_frame = draw_hand_landmarks(display_frame, results)
                
                # Extract hand region and get bounding box
                hand_img, hand_detected, bbox = extract_hand_region(frame, results)
                if hand_detected:
                    display_frame = draw_hand_box(display_frame, *bbox)
                    status_color = (0, 255, 0)  # Green if hand detected
                    status_text = "Hand detected - ready to capture"
                else:
                    status_color = (0, 0, 255)  # Red if no hand detected
                    status_text = "No hand detected"
                
                # Add instructions and status
                cv2.putText(display_frame, f"Show '{gesture_name}' gesture (Image {count+1}/{num_images})", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, status_text,
                          (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'n' for next, 'q' to quit", 
                          (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show countdown if hand is detected
                if hand_detected:
                    current_time = time.time()
                    if not 'start_time' in locals():
                        start_time = current_time
                    
                    elapsed_time = current_time - start_time
                    if elapsed_time < waiting_time:
                        countdown = waiting_time - int(elapsed_time)
                        cv2.putText(display_frame, f"Capturing in: {countdown}", 
                                  (frame.shape[1]//2-100, frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Auto-capture
                        success, message = save_image_with_hand(
                            frame, gesture_dir, gesture_name, len(os.listdir(gesture_dir)), results
                        )
                        if success:
                            count += 1
                            # Save a screenshot for documentation
                            if debug:
                                save_screenshot(display_frame)
                        start_time = current_time + 1  # Add a delay before next capture
                else:
                    if 'start_time' in locals():
                        del start_time  # Reset timer if hand is lost
                
                # Display the frame
                cv2.imshow('Capture Training Images', display_frame)
                
                # Display debug window if enabled
                if debug and hand_detected:
                    if hand_img is not None:
                        cv2.imshow('Hand Region', hand_img)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("Quitting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    print(f"Moving to next gesture. Captured {count} images for {gesture_name}.")
                    break
                elif key == ord('c') and hand_detected:
                    # Manual capture
                    success, message = save_image_with_hand(
                        frame, gesture_dir, gesture_name, len(os.listdir(gesture_dir)), results
                    )
                    if success:
                        count += 1
                        if debug:
                            save_screenshot(display_frame)
                    else:
                        print(f"Failed to save: {message}")
                    time.sleep(1)  # Short delay to prevent multiple captures
            
            print(f"Completed capturing {count} images for {gesture_name}.")
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    print("\nImage capture completed. You can now run model_training.py to train the model.")

def count_images():
    """Count the number of images in each class directory"""
    try:
        rock_dir, paper_dir, scissors_dir = create_directories()
        
        rock_count = len([f for f in os.listdir(rock_dir) if f.endswith('.jpg')])
        paper_count = len([f for f in os.listdir(paper_dir) if f.endswith('.jpg')])
        scissors_count = len([f for f in os.listdir(scissors_dir) if f.endswith('.jpg')])
        
        print("\nCurrent dataset statistics:")
        print(f"Rock images: {rock_count}")
        print(f"Paper images: {paper_count}")
        print(f"Scissors images: {scissors_count}")
        print(f"Total images: {rock_count + paper_count + scissors_count}")
        
        if rock_count < 20 or paper_count < 20 or scissors_count < 20:
            print("\nNote: It's recommended to have at least 20 images per class for good model performance.")
            
    except Exception as e:
        print(f"Error counting images: {e}")

if __name__ == "__main__":
    try:
        # Show current dataset statistics
        count_images()
        
        # Ask if user wants to add more images
        print("\nDo you want to capture more training images?")
        print("1. Yes, with debugging (shows hand region)")
        print("2. Yes, normal mode")
        print("3. No, exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            capture_training_images(debug=True)
        elif choice == '2':
            capture_training_images(debug=False)
        else:
            print("Exiting...")
            
    except Exception as e:
        print(f"Error: {e}") 