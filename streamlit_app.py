import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
from PIL import Image
import time
from utils import extract_hand_features, preprocess_landmarks, draw_hand_landmarks

st.set_page_config(
    page_title="Rock Paper Scissors AI Game",
    page_icon="✂️",
    layout="wide"
)

# CSS to improve the UI
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #ff9d00;
}
.game-area {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
}
.score-box {
    background-color: #3d4654;
    padding: 10px 20px;
    border-radius: 5px;
    margin: 10px 0;
    font-weight: bold;
    font-size: 1.2em;
}
.user-score {
    color: #00ff8c;
}
.computer-score {
    color: #ff5b77;
}
.stButton button {
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize the game
class StreamlitRockPaperScissorsGame:
    def __init__(self):
        self.class_labels = ['rock', 'paper', 'scissors']
        
        # Initialize session state if not already done
        if 'user_score' not in st.session_state:
            st.session_state.user_score = 0
        if 'computer_score' not in st.session_state:
            st.session_state.computer_score = 0
        if 'round_count' not in st.session_state:
            st.session_state.round_count = 0
        if 'max_rounds' not in st.session_state:
            st.session_state.max_rounds = 5
        if 'result' not in st.session_state:
            st.session_state.result = None
        if 'user_move' not in st.session_state:
            st.session_state.user_move = None
        if 'computer_move' not in st.session_state:
            st.session_state.computer_move = None
        if 'game_state' not in st.session_state:
            st.session_state.game_state = "welcome"
        if 'confidence' not in st.session_state:
            st.session_state.confidence = None
        if 'last_image' not in st.session_state:
            st.session_state.last_image = None
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model for gesture recognition"""
        model_path = "models/rps_model_latest.pkl"
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return False
            
            model = joblib.load(model_path)
            
            # Check model type and extract if needed
            if isinstance(model, dict) and 'model' in model:
                self.model = model['model']
            else:
                self.model = model
                
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    
    def predict_user_move(self, image):
        """Predict the user's move from the uploaded or captured image"""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process the image with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Draw hand landmarks for visualization
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        # Extract features and predict if hand detected
        if results.multi_hand_landmarks:
            try:
                # Extract features (adaptation needed based on your model training)
                img_resized = cv2.resize(image, (64, 64))
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                features = img_gray.flatten()
                
                # Make prediction
                prediction = self.model.predict([features])[0]
                
                # Handle different prediction types
                if isinstance(prediction, (int, np.integer)):
                    move = self.class_labels[prediction]
                else:
                    move = str(prediction)
                
                # Get confidence if available
                confidence = 0.7  # Default confidence
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba([features])[0]
                    confidence = float(np.max(probs))
                
                return move, confidence, annotated_image, True
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return "unknown", 0.0, annotated_image, False
        else:
            return "unknown", 0.0, annotated_image, False
    
    def get_computer_move(self):
        """Generate a random move for the computer"""
        import random
        return random.choice(self.class_labels)
    
    def determine_winner(self, user_move, computer_move):
        """Determine the winner of the round"""
        if user_move == computer_move:
            return "It's a tie!"
        
        if user_move == "rock":
            return "You win!" if computer_move == "scissors" else "Computer wins!"
        
        if user_move == "paper":
            return "You win!" if computer_move == "rock" else "Computer wins!"
        
        if user_move == "scissors":
            return "You win!" if computer_move == "paper" else "Computer wins!"
        
        return "Invalid move"
    
    def reset_game(self):
        """Reset the game state"""
        st.session_state.user_score = 0
        st.session_state.computer_score = 0
        st.session_state.round_count = 0
        st.session_state.game_state = "welcome"
        st.session_state.result = None
        st.session_state.user_move = None
        st.session_state.computer_move = None
    
    def next_round(self):
        """Move to the next round"""
        st.session_state.game_state = "playing"
        st.session_state.result = None
        st.session_state.user_move = None
        st.session_state.computer_move = None

# Main app
def main():
    # Title
    st.title("Rock Paper Scissors AI Game")
    st.markdown("Play Rock Paper Scissors against an AI using your webcam or uploaded images!")
    
    # Initialize game
    game = StreamlitRockPaperScissorsGame()
    
    # Game UI based on state
    if st.session_state.game_state == "welcome":
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("## Welcome to Rock Paper Scissors!")
            st.markdown("""
            This game uses computer vision and machine learning to recognize your hand gestures.
            
            ### How to Play:
            1. Choose to use your webcam or upload an image
            2. Show your hand gesture (rock, paper, scissors)
            3. The AI will predict your move and play against you
            4. First to 5 points wins!
            
            ### Controls:
            - **Rock**: Make a fist
            - **Paper**: Show an open palm
            - **Scissors**: Extend your index and middle fingers
            """)
            
            if st.button("Start Game", key="start_game"):
                st.session_state.game_state = "playing"
                st.rerun()
        
        with col2:
            # Show sample image
            if os.path.exists("screenshots/live_gameplay.png"):
                st.image("screenshots/live_gameplay.png", caption="Game in action")
    
    elif st.session_state.game_state == "playing":
        # Display scoreboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='score-box user-score'>You: {st.session_state.user_score}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='score-box'>Round: {st.session_state.round_count}/{st.session_state.max_rounds}</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='score-box computer-score'>Computer: {st.session_state.computer_score}</div>", unsafe_allow_html=True)
        
        # Input options
        st.markdown("### Show your move!")
        input_type = st.radio("Choose input method:", ["Upload Image", "Keyboard Input"], horizontal=True)
        
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader("Upload your hand gesture", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Process the image
                image = Image.open(uploaded_file)
                st.session_state.last_image = image  # Store for display
                st.image(image, caption="Your uploaded image", width=300)
                
                if st.button("Make Prediction"):
                    move, confidence, annotated_image, success = game.predict_user_move(image)
                    
                    if success and move != "unknown":
                        # Get computer move
                        computer_move = game.get_computer_move()
                        
                        # Determine winner
                        result = game.determine_winner(move, computer_move)
                        
                        # Update scores
                        if result == "You win!":
                            st.session_state.user_score += 1
                        elif result == "Computer wins!":
                            st.session_state.computer_score += 1
                        
                        # Update state
                        st.session_state.user_move = move
                        st.session_state.computer_move = computer_move
                        st.session_state.result = result
                        st.session_state.confidence = confidence
                        st.session_state.round_count += 1
                        st.session_state.game_state = "result"
                        st.session_state.last_image = annotated_image
                        
                        st.rerun()
                    else:
                        st.error("No hand detected or unable to predict gesture. Please try again.")
        
        else:  # Keyboard Input
            st.markdown("### Select your move using the keyboard:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("ROCK 👊"):
                    st.session_state.user_move = "rock"
                    st.session_state.confidence = 1.0
                    st.session_state.computer_move = game.get_computer_move()
                    st.session_state.result = game.determine_winner(st.session_state.user_move, st.session_state.computer_move)
                    
                    if st.session_state.result == "You win!":
                        st.session_state.user_score += 1
                    elif st.session_state.result == "Computer wins!":
                        st.session_state.computer_score += 1
                    
                    st.session_state.round_count += 1
                    st.session_state.game_state = "result"
                    st.rerun()
            
            with col2:
                if st.button("PAPER ✋"):
                    st.session_state.user_move = "paper"
                    st.session_state.confidence = 1.0
                    st.session_state.computer_move = game.get_computer_move()
                    st.session_state.result = game.determine_winner(st.session_state.user_move, st.session_state.computer_move)
                    
                    if st.session_state.result == "You win!":
                        st.session_state.user_score += 1
                    elif st.session_state.result == "Computer wins!":
                        st.session_state.computer_score += 1
                    
                    st.session_state.round_count += 1
                    st.session_state.game_state = "result"
                    st.rerun()
            
            with col3:
                if st.button("SCISSORS ✌️"):
                    st.session_state.user_move = "scissors"
                    st.session_state.confidence = 1.0
                    st.session_state.computer_move = game.get_computer_move()
                    st.session_state.result = game.determine_winner(st.session_state.user_move, st.session_state.computer_move)
                    
                    if st.session_state.result == "You win!":
                        st.session_state.user_score += 1
                    elif st.session_state.result == "Computer wins!":
                        st.session_state.computer_score += 1
                    
                    st.session_state.round_count += 1
                    st.session_state.game_state = "result"
                    st.rerun()
    
    elif st.session_state.game_state == "result":
        # Display the result
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("## Round Result")
            
            # Display moves and result
            st.markdown(f"### Your move: {st.session_state.user_move.upper()}")
            st.markdown(f"### Computer's move: {st.session_state.computer_move.upper()}")
            
            # Display result with appropriate color
            result_color = "white"
            if st.session_state.result == "You win!":
                result_color = "green"
            elif st.session_state.result == "Computer wins!":
                result_color = "red"
            
            st.markdown(f"<h2 style='color:{result_color};'>{st.session_state.result}</h2>", unsafe_allow_html=True)
            
            # Display confidence if it exists
            if st.session_state.confidence is not None:
                st.markdown(f"Prediction confidence: {st.session_state.confidence:.2f}")
            
            # Display current score
            st.markdown(f"### Score: You {st.session_state.user_score} - {st.session_state.computer_score} Computer")
            
            # Check if game over
            if st.session_state.round_count >= st.session_state.max_rounds:
                st.session_state.game_state = "gameover"
                st.rerun()
            else:
                if st.button("Next Round"):
                    game.next_round()
                    st.rerun()
        
        with col2:
            # Show the image with annotation if it exists
            if st.session_state.last_image is not None:
                st.image(st.session_state.last_image, caption="Your move", width=300)
    
    elif st.session_state.game_state == "gameover":
        st.markdown("## Game Over!")
        
        # Determine the final winner
        if st.session_state.user_score > st.session_state.computer_score:
            st.success("### 🎉 You Win! 🎉")
        elif st.session_state.user_score < st.session_state.computer_score:
            st.error("### Computer Wins!")
        else:
            st.info("### It's a Tie!")
        
        st.markdown(f"### Final Score: You {st.session_state.user_score} - {st.session_state.computer_score} Computer")
        
        if st.button("Play Again"):
            game.reset_game()
            st.rerun()
    
    # Always show these buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Game"):
            game.reset_game()
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with StreamLit and MediaPipe • [GitHub Repository](https://github.com/yash240602/Rock-Paper-Scissor)")

if __name__ == "__main__":
    main() 