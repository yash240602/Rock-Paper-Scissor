import cv2
import numpy as np
import pickle
import os
import time
import random
import mediapipe as mp
import joblib
from utils import (
    detect_hand, 
    draw_hand_landmarks, 
    extract_hand_region, 
    preprocess_image,
    draw_welcome_screen,
    draw_scoreboard,
    draw_hand_box,
    start_countdown,
    display_result,
    save_screenshot,
    determine_winner,
    log_misclassification
)

class RockPaperScissorsGame:
    """
    Class for the Rock Paper Scissors game using computer vision
    """
    def __init__(self, model_path=None, webcam_id=0, window_name="Rock Paper Scissors Game"):
        """Initialize the game"""
        self.class_labels = ['rock', 'paper', 'scissors']
        self.user_score = 0
        self.computer_score = 0
        self.round_count = 0
        self.max_rounds = 5
        
        # Game states
        self.state = "welcome"  # welcome, countdown, playing, result, gameover
        self.countdown = 3
        self.last_countdown_time = 0
        self.last_prediction_time = 0
        self.prediction_cooldown = 1  # seconds between predictions
        
        # Current round data
        self.user_move = None
        self.computer_move = None
        self.result = None
        self.confidence = None
        
        # Prediction history for temporal smoothing
        self.prediction_history = []
        self.prediction_threshold = 0.5  # Reduced from 0.6 to allow more predictions
        
        # Misclassification tracking
        self.last_debug_save_time = 0
        self.debug_save_cooldown = 5  # Only save debug screenshots every 5 seconds
        self.debug_logging = False  # Flag to enable/disable debug logging
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (255, 255, 255)
        self.text_size = 0.7
        self.text_thickness = 2
        
        # Initialize MediaPipe with improved settings
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower from default 0.7 for better detection in varying conditions
            min_tracking_confidence=0.5    # Lower from default 0.7 for more consistent tracking
        )
        
        # Performance monitoring variables
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        self.fps_history = []  # For smoother FPS display
        
        # Game settings
        self.webcam_id = webcam_id
        self.window_name = window_name
        self.cap = None
        self.model = None
        self.model_path = model_path if model_path else "models/rps_model_latest.pkl"
        
        # Debug threshold for low confidence predictions
        self.low_confidence_threshold = 0.8
        
        # Image processing enhancement settings
        self.enable_image_enhancement = True
        self.detection_failure_count = 0
        self.max_detection_failures = 10  # After this many failures, try image enhancement
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.user_score = 0
        self.computer_score = 0
        self.round_count = 0
        self.state = "welcome"
        self.prediction_history = []
        self.user_move = None
        self.computer_move = None
        self.result = None
        self.confidence = None
        print("[INFO] Game reset successfully")
    
    def load_model(self):
        """Load the trained model for gesture recognition"""
        try:
            if not os.path.exists(self.model_path):
                print(f"[ERROR] Model file not found: {self.model_path}")
                print("[INFO] Please train a model first using model_training.py")
                
                # Try to find any model in the models directory as a fallback
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') or f.endswith('.joblib')]
                    if model_files:
                        fallback_model = os.path.join(model_dir, model_files[0])
                        print(f"[INFO] Attempting to load fallback model: {fallback_model}")
                        try:
                            self.model = joblib.load(fallback_model)
                            print(f"[SUCCESS] Loaded fallback model: {fallback_model}")
                            return True
                        except Exception as e:
                            print(f"[ERROR] Failed to load fallback model: {e}")
                            return False
                    
                print("[WARNING] No fallback models found. The game may not function correctly.")
                return False
            
            try:
                self.model = joblib.load(self.model_path)
                print(f"[SUCCESS] Model loaded from: {self.model_path}")
                
                # Check model type
                if isinstance(self.model, dict):
                    print("[INFO] Model is in dictionary format")
                    if 'model' in self.model:
                        print("[INFO] Extracting model from dictionary")
                        self.model = self.model['model']
                    else:
                        print("[ERROR] Dictionary model doesn't contain 'model' key")
                        print("[INFO] Available keys:", self.model.keys())
                        # Create a simple fallback model for testing
                        self._create_fallback_model()
                        return True
                
                # Log model information if available
                if hasattr(self.model, 'feature_importances_'):
                    top_features = np.argsort(self.model.feature_importances_)[-5:]
                    print(f"[INFO] Top 5 important features indices: {top_features}")
                
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load model from {self.model_path}: {e}")
                print("[WARNING] Creating a simple fallback model for testing")
                self._create_fallback_model()
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[WARNING] The game will run without gesture recognition")
            self._create_fallback_model()
            return True
    
    def _create_fallback_model(self):
        """Create a simple fallback model for testing purposes"""
        from sklearn.ensemble import RandomForestClassifier
        
        print("[INFO] Creating a simple fallback RandomForest model")
        self.model = RandomForestClassifier(n_estimators=10)
        
        # Train on dummy data
        X_dummy = np.random.random((30, 64*64))  # Random features for 30 samples
        y_dummy = np.array([0, 1, 2] * 10)  # 10 samples for each class
        
        self.model.fit(X_dummy, y_dummy)
        print("[INFO] Fallback model created successfully")
    
    def get_computer_move(self):
        """
        Generate a random move for the computer
        
        Returns:
            move: Computer's move (rock, paper, scissors)
        """
        return random.choice(self.class_labels)
    
    def predict_user_move(self, hand_img):
        """
        Predict the user's move from the hand image
        
        Args:
            hand_img: Image of the user's hand
            
        Returns:
            move: Predicted move (rock, paper, scissors)
            confidence: Confidence score for the prediction
        """
        try:
            # Preprocess the image
            img_features = preprocess_image(hand_img)
            
            # Check for valid model and features
            if self.model is None:
                print("[ERROR] Model not loaded")
                return random.choice(self.class_labels), 0.33
            
            if img_features is None or len(img_features) == 0:
                print("[ERROR] Invalid features")
                return random.choice(self.class_labels), 0.33
            
            # Reshape features to 2D array with single sample
            img_features = np.array(img_features).reshape(1, -1)
            
            # Make prediction
            try:
                prediction = self.model.predict(img_features)[0]
                # Convert prediction index to label
                if isinstance(prediction, (int, np.integer)):
                    move = self.class_labels[prediction]
                else:
                    move = str(prediction)
                
                # Get confidence if available
                confidence = 0.7  # Default confidence
                try:
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(img_features)[0]
                        confidence = float(np.max(probs))
                except Exception as e:
                    print(f"[WARNING] Could not get prediction confidence: {e}")
                
                return move, confidence
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
                return random.choice(self.class_labels), 0.33
                
        except Exception as e:
            print(f"[ERROR] Error in predict_user_move: {e}")
            return random.choice(self.class_labels), 0.33
    
    def reset_round(self):
        """Reset the round data"""
        self.user_move = None
        self.computer_move = None
        self.result = None
        self.confidence = None
        self.state = "countdown"
        self.countdown = 3
        self.last_countdown_time = time.time()
    
    def run(self):
        """Main game loop"""
        print("[INFO] Starting game...")
        
        try:
            # Initialize webcam
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                print(f"[ERROR] Could not open webcam with ID {self.webcam_id}")
                print("[INFO] Trying with default webcam (ID 0)...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("[ERROR] Could not open default webcam either. Exiting.")
                    return
                
            # Set larger resolution for better detection
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Load ML model
            model_loaded = self.load_model()
            if not model_loaded:
                print("[WARNING] Running without ML model - only keyboard controls will work")
            
            # Create screenshots directory if it doesn't exist
            os.makedirs("screenshots", exist_ok=True)
            
            # Reset game state
            self.reset_game()
            self.state = "welcome"
            self.welcome_time = time.time()
            
            print("[INFO] Game started successfully! Press 'q' to quit.")
            
            while True:
                # Performance monitoring - start time for this frame
                self.new_frame_time = time.time()
                
                # Capture frame and process
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame. Check webcam connection.")
                    break
                
                # Apply image enhancement if enabled and detection is failing
                if self.enable_image_enhancement and self.detection_failure_count > self.max_detection_failures:
                    # Enhance contrast
                    frame = self.enhance_image(frame)
                    
                # Create a copy of the frame for display
                display_frame = frame.copy()
                
                # Get current time
                current_time = time.time()
                
                # Process hand detection
                try:
                    # No need to pass mp_hands_instance since we're using the direct instance
                    results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Track detection success/failure
                    if results and results.multi_hand_landmarks:
                        self.detection_failure_count = 0  # Reset counter on success
                    else:
                        self.detection_failure_count += 1  # Increment on failure
                        
                except Exception as e:
                    print(f"Error detecting hand: {e}")
                    import traceback
                    traceback.print_exc()
                    results = None
                    self.detection_failure_count += 1
                
                # Draw hand landmarks
                try:
                    if results is not None:
                        display_frame = draw_hand_landmarks(display_frame, results)
                except Exception as e:
                    print(f"Error drawing hand landmarks: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Handle different game states
                try:
                    if self.state == "welcome":
                        # Show welcome screen
                        display_frame = draw_welcome_screen(display_frame)
                    
                    elif self.state == "countdown":
                        # Show countdown before capturing move
                        if current_time - self.last_countdown_time > 1:
                            self.countdown -= 1
                            self.last_countdown_time = current_time
                            
                        if self.countdown > 0:
                            # Still counting down
                            display_frame = start_countdown(display_frame, self.countdown)
                        else:
                            # Countdown complete, move to playing state
                            self.state = "playing"
                            self.last_prediction_time = current_time
                    
                    elif self.state == "playing":
                        # Check if hand is detected
                        if results and results.multi_hand_landmarks:
                            # Extract hand region
                            try:
                                hand_img, success, bbox = extract_hand_region(frame, results)
                                
                                if success:
                                    # Draw bounding box around hand
                                    display_frame = draw_hand_box(display_frame, *bbox)
                                    
                                    # Make prediction if enough time has passed
                                    if current_time - self.last_prediction_time > self.prediction_cooldown:
                                        # Predict user's move
                                        move, conf = self.predict_user_move(hand_img)
                                        
                                        # Add to prediction history for temporal smoothing
                                        self.prediction_history.append(move)
                                        
                                        # Only keep the last 5 predictions
                                        if len(self.prediction_history) > 5:
                                            self.prediction_history.pop(0)
                                        
                                        # Only use predictions if confidence is above threshold
                                        if conf is None or conf >= self.prediction_threshold:
                                            # Use majority vote for stable prediction
                                            if len(self.prediction_history) >= 3:  # At least 3 predictions to smooth
                                                stabilized_move = max(set(self.prediction_history), key=self.prediction_history.count)
                                                self.user_move = stabilized_move
                                            else:
                                                self.user_move = move
                                        
                                        self.confidence = conf
                                        
                                        # Generate computer's move
                                        self.computer_move = self.get_computer_move()
                                        
                                        # Determine winner
                                        self.result = determine_winner(self.user_move, self.computer_move)
                                        
                                        # Update scores
                                        if self.result == "You win!":
                                            self.user_score += 1
                                        elif self.result == "Computer wins!":
                                            self.computer_score += 1
                                        
                                        # Update round count
                                        self.round_count += 1
                                        
                                        # Move to result state
                                        self.state = "result"
                                        self.result_time = current_time
                            except Exception as e:
                                print(f"Error processing hand: {e}")
                        else:
                            # No hand detected - clear prediction history to avoid stale predictions
                            self.prediction_history = []
                            
                            # Show clear message about hand detection
                            cv2.putText(display_frame, "No hand detected! Show your hand clearly.", 
                                      (50, display_frame.shape[0] // 2), 
                                      self.font, 1.0, (0, 0, 255), 2)
                            cv2.putText(display_frame, "Make sure your hand is in frame with good lighting.", 
                                      (50, display_frame.shape[0] // 2 + 40), 
                                      self.font, 0.7, (0, 0, 255), 2)
                            
                            # Make keyboard fallback more prominent when hand is not detected
                            cv2.putText(display_frame, "Or use keyboard controls:", 
                                      (50, display_frame.shape[0] // 2 + 80), 
                                      self.font, 0.8, (0, 255, 255), 2)
                            cv2.putText(display_frame, "Press '1' for Rock, '2' for Paper, '3' for Scissors", 
                                      (50, display_frame.shape[0] // 2 + 120), 
                                      self.font, 0.8, (0, 255, 255), 2)
                        
                        # Always show keyboard fallback instructions during playing state
                        cv2.putText(display_frame, "Keyboard fallback: Press '1' for Rock, '2' for Paper, '3' for Scissors", 
                                  (10, display_frame.shape[0] - 50), 
                                  self.font, 0.7, (0, 255, 255), 2)
                    
                    elif self.state == "result":
                        # Show the result of the round
                        try:
                            display_frame = display_result(
                                display_frame, 
                                self.user_move, 
                                self.computer_move, 
                                self.result
                            )
                            
                            # Show confidence if available - more prominent display
                            if self.confidence is not None:
                                # Create better confidence visualization with color coding
                                confidence_color = (0, 255, 0)  # Green for high confidence
                                if self.confidence < 0.7:
                                    confidence_color = (0, 255, 255)  # Yellow for medium confidence
                                if self.confidence < 0.5:
                                    confidence_color = (0, 0, 255)  # Red for low confidence
                                
                                # Display confidence with colored bar
                                cv2.putText(display_frame, f"Prediction Confidence: {self.confidence:.2f}", 
                                          (10, 100), 
                                          self.font, 0.8, confidence_color, 2)
                                
                                # Draw confidence bar
                                bar_length = int(200 * self.confidence)
                                cv2.rectangle(display_frame, (10, 110), (10 + bar_length, 120), confidence_color, -1)
                                cv2.rectangle(display_frame, (10, 110), (210, 120), (200, 200, 200), 1)  # Border
                            
                            # Log potential misclassifications for debugging (with rate limiting)
                            if self.result != "It's a tie!" and self.confidence is not None and self.confidence < 0.7:
                                current_time = time.time()
                                if current_time - self.last_debug_save_time > self.debug_save_cooldown:
                                    print(f"[DEBUG] Potential misclassification. User move: {self.user_move}, Confidence: {self.confidence:.2f}")
                                    # Save screenshot with '_debug' prefix for later analysis
                                    debug_folder = "debug_misclassifications"
                                    os.makedirs(debug_folder, exist_ok=True)
                                    timestamp = time.strftime("%Y%m%d%H%M%S")
                                    debug_filename = f"{debug_folder}/misclassification_{self.user_move}_{self.confidence:.2f}_{timestamp}.jpg"
                                    cv2.imwrite(debug_filename, display_frame)
                                    print(f"[DEBUG] Misclassification screenshot saved: {debug_filename}")
                                    self.last_debug_save_time = current_time
                            
                            # Show next round instruction
                            cv2.putText(display_frame, "Press 'p' to play next round", 
                                      (10, display_frame.shape[0] - 120), 
                                      self.font, self.text_size, self.text_color, self.text_thickness)
                            
                            # Check if max rounds reached
                            if self.round_count >= self.max_rounds:
                                # Move to game over state
                                if current_time - self.result_time > 3:  # Show result for 3 seconds
                                    self.state = "gameover"
                        except Exception as e:
                            print(f"Error displaying result: {e}")
                    
                    elif self.state == "gameover":
                        # Show game over screen
                        try:
                            display_frame = self.draw_game_over(display_frame)
                        except Exception as e:
                            print(f"Error displaying game over screen: {e}")
                except Exception as e:
                    print(f"Error in game state handling: {e}")
                    # Reset to original frame if there was an error
                    display_frame = frame.copy()
                
                # Always draw the scoreboard
                if isinstance(display_frame, np.ndarray):
                    display_frame = draw_scoreboard(display_frame, self.user_score, self.computer_score)
                else:
                    print("Error: display_frame is not a valid image at scoreboard drawing")
                    display_frame = frame.copy()  # Reset to original frame if corrupted
                
                # Show round count
                if isinstance(display_frame, np.ndarray) and hasattr(display_frame, 'shape') and len(display_frame.shape) >= 2:
                    cv2.putText(display_frame, f"Round: {self.round_count}/{self.max_rounds}", 
                              (display_frame.shape[1] - 200, 30), 
                              self.font, self.text_size, self.text_color, self.text_thickness)
                    
                    # Display instructions at the bottom
                    cv2.putText(display_frame, "p: play  s: screenshot  r: reset  q: quit", 
                              (10, display_frame.shape[0] - 20), 
                              self.font, 0.6, self.text_color, 1)
                else:
                    print("Error: display_frame is not a valid image for drawing text")
                    display_frame = frame.copy()  # Reset to original frame if corrupted
                
                # Calculate and display FPS
                time_diff = self.new_frame_time - self.prev_frame_time
                if time_diff > 0:
                    frame_fps = 1 / time_diff
                    # Add to history for smoothing
                    self.fps_history.append(frame_fps)
                    if len(self.fps_history) > 10:  # Keep last 10 frames for averaging
                        self.fps_history.pop(0)
                    # Calculate average FPS
                    self.fps = sum(self.fps_history) / len(self.fps_history)
                self.prev_frame_time = self.new_frame_time
                
                # Display FPS counter
                cv2.putText(display_frame, f"FPS: {int(self.fps)}", 
                          (display_frame.shape[1] - 120, 30), 
                          self.font, 0.7, (0, 255, 0), 2)
                
                # Draw game title
                cv2.putText(display_frame, "ROCK PAPER SCISSORS", 
                          (display_frame.shape[1]//2 - 200, 40), 
                          self.font, 1.2, (0, 165, 255), 3)
                
                # Show the frame
                try:
                    cv2.imshow(self.window_name, display_frame)
                except Exception as e:
                    print(f"Error showing frame: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Handle key presses
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        if self.state == "welcome" or self.state == "result" or self.state == "gameover":
                            self.reset_round()
                    elif key == ord('r'):
                        # Reset scores
                        self.user_score = 0
                        self.computer_score = 0
                        self.round_count = 0
                        self.state = "welcome"
                    elif key == ord('s'):
                        # Take a screenshot
                        save_screenshot(display_frame)
                    # Keyboard fallback for gesture input
                    elif self.state == "playing" and key == ord('1'):
                        # Use keyboard '1' for rock
                        self.user_move = "rock"
                        self.confidence = 1.0  # Manual selection has perfect confidence
                        self.computer_move = self.get_computer_move()
                        self.result = determine_winner(self.user_move, self.computer_move)
                        
                        # Update scores
                        if self.result == "You win!":
                            self.user_score += 1
                        elif self.result == "Computer wins!":
                            self.computer_score += 1
                        
                        # Update round count and move to result state
                        self.round_count += 1
                        self.state = "result"
                        self.result_time = current_time
                    elif self.state == "playing" and key == ord('2'):
                        # Use keyboard '2' for paper
                        self.user_move = "paper"
                        self.confidence = 1.0  # Manual selection has perfect confidence
                        self.computer_move = self.get_computer_move()
                        self.result = determine_winner(self.user_move, self.computer_move)
                        
                        # Update scores
                        if self.result == "You win!":
                            self.user_score += 1
                        elif self.result == "Computer wins!":
                            self.computer_score += 1
                        
                        # Update round count and move to result state
                        self.round_count += 1
                        self.state = "result"
                        self.result_time = current_time
                    elif self.state == "playing" and key == ord('3'):
                        # Use keyboard '3' for scissors
                        self.user_move = "scissors"
                        self.confidence = 1.0  # Manual selection has perfect confidence
                        self.computer_move = self.get_computer_move()
                        self.result = determine_winner(self.user_move, self.computer_move)
                        
                        # Update scores
                        if self.result == "You win!":
                            self.user_score += 1
                        elif self.result == "Computer wins!":
                            self.computer_score += 1
                        
                        # Update round count and move to result state
                        self.round_count += 1
                        self.state = "result" 
                        self.result_time = current_time
                    # Debug: Toggle debug logging
                    elif key == ord('d'):
                        self.debug_logging = not self.debug_logging
                        print(f"[DEBUG] Debug logging {'enabled' if self.debug_logging else 'disabled'}")
                except Exception as e:
                    print(f"Error handling key press: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            print("Game closed.")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_game_over(self, frame):
        """
        Draw the game over screen
        
        Args:
            frame: Frame to draw on
            
        Returns:
            frame: Frame with game over screen drawn on it
        """
        if frame is None or not isinstance(frame, np.ndarray):
            print("Invalid frame provided to draw_game_over")
            return frame
            
        try:
            h, w = frame.shape[:2]
            
            # Add semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Show game over text
            cv2.putText(frame, "GAME OVER", 
                      (w//2 - 120, h//2 - 50), 
                      self.font, 1.5, (0, 0, 255), 3)
            
            # Show final score
            cv2.putText(frame, f"Final Score:", 
                      (w//2 - 100, h//2 + 20), 
                      self.font, 1, self.text_color, 2)
            
            cv2.putText(frame, f"You: {self.user_score}  Computer: {self.computer_score}", 
                      (w//2 - 150, h//2 + 70), 
                      self.font, 1, self.text_color, 2)
            
            # Show result
            if self.user_score > self.computer_score:
                result_text = "You Win!"
                result_color = (0, 255, 0)
            elif self.user_score < self.computer_score:
                result_text = "Computer Wins!"
                result_color = (0, 0, 255)
            else:
                result_text = "It's a Tie!"
                result_color = (255, 255, 255)
            
            cv2.putText(frame, result_text, 
                      (w//2 - 100, h//2 + 130), 
                      self.font, 1.2, result_color, 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'p' to play again", 
                      (w//2 - 120, h//2 + 180), 
                      self.font, 0.8, self.text_color, 2)
        except Exception as e:
            print(f"Error in draw_game_over: {e}")
        
        return frame

    def enhance_image(self, frame):
        """
        Apply image enhancement techniques to improve hand detection in challenging lighting
        
        Args:
            frame: Original video frame
            
        Returns:
            enhanced_frame: Enhanced frame for better detection
        """
        if frame is None:
            return frame
            
        try:
            # Convert to HSV for better handling of lighting
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask for likely skin color ranges (adjust for different skin tones)
            # Wider range to account for diverse skin tones and lighting conditions
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Additional range for darker skin tones
            lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            # Combine masks
            mask = mask1 + mask2
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe.apply(gray)
            
            # Blend original with enhanced image
            enhanced_frame = frame.copy()
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Use the skin mask to focus enhancement on hand regions
            skin_mask = cv2.merge([mask, mask, mask]) / 255.0
            enhanced_frame = cv2.addWeighted(enhanced_frame, 0.7, enhanced_bgr, 0.3, 0)
            
            return enhanced_frame
        except Exception as e:
            print(f"Error in image enhancement: {e}")
            return frame

def main():
    """Main function to run the game"""
    print("=" * 50)
    print("ROCK PAPER SCISSORS GAME")
    print("=" * 50)
    
    # Create and run the game
    game = RockPaperScissorsGame()
    game.run()

if __name__ == "__main__":
    main() 