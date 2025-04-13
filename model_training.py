import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import mediapipe as mp
from utils import extract_hand_region, preprocess_image, augment_image

class RockPaperScissorsModel:
    """
    Class for training and evaluating a rock-paper-scissors gesture recognition model
    """
    def __init__(self, data_dir="data", model_dir="models"):
        """
        Initialize the model trainer
        
        Args:
            data_dir: Directory containing class subdirectories with training images
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.class_names = ["rock", "paper", "scissors"]
        self.mp_hands = mp.solutions.hands
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"Initialized RockPaperScissorsModel with data_dir={data_dir}, model_dir={model_dir}")
    
    def load_and_preprocess_data(self, use_augmentation=True, use_cropped=True):
        """
        Load images from the data directory, preprocess them, and prepare for training
        
        Args:
            use_augmentation: Whether to use data augmentation
            use_cropped: Whether to use cropped hand images instead of full frames
            
        Returns:
            X, y: Feature matrix and target labels
        """
        print("Loading and preprocessing data...")
        X = []
        y = []
        class_counts = {}
        augmented_counts = {}
        
        # Initialize hands detector for processing images without detected hands
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            
            # Process each class directory
            for class_name in self.class_names:
                class_dir = os.path.join(self.data_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: Directory {class_dir} does not exist!")
                    continue
                
                class_counts[class_name] = 0
                augmented_counts[class_name] = 0
                
                print(f"Processing {class_name} images from {class_dir}")
                
                # Get all image files
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                              and not f.endswith('_cropped.jpg')]
                
                for img_file in image_files:
                    # Check if we should use the cropped version
                    img_path = os.path.join(class_dir, img_file)
                    cropped_img_path = os.path.join(class_dir, img_file.replace('.jpg', '_cropped.jpg'))
                    
                    if use_cropped and os.path.exists(cropped_img_path):
                        # Use the cropped version
                        img = cv2.imread(cropped_img_path)
                    else:
                        # Use the full frame
                        img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Check if we need to extract hand region for non-cropped images
                    if not use_cropped:
                        # Convert to RGB for MediaPipe
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Process with MediaPipe to get hand region
                        results = hands.process(img_rgb)
                        
                        if results.multi_hand_landmarks:
                            # Extract hand region
                            hand_img, success, _ = extract_hand_region(img, results)
                            if success:
                                img = hand_img
                            else:
                                print(f"Warning: Failed to extract hand region from {img_path}")
                    
                    # Preprocess the image
                    try:
                        img_features = preprocess_image(img)
                        X.append(img_features)
                        y.append(class_name)
                        class_counts[class_name] += 1
                        
                        # Apply data augmentation if enabled
                        if use_augmentation:
                            augmented_images = augment_image(img)
                            # Skip the first one as it's the original
                            for aug_img in augmented_images[1:]:
                                aug_features = preprocess_image(aug_img)
                                X.append(aug_features)
                                y.append(class_name)
                                augmented_counts[class_name] += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Print dataset statistics
        print("\nDataset statistics:")
        total_original = sum(class_counts.values())
        total_augmented = sum(augmented_counts.values())
        
        for class_name in self.class_names:
            orig = class_counts.get(class_name, 0)
            aug = augmented_counts.get(class_name, 0)
            print(f"{class_name}: {orig} original + {aug} augmented = {orig + aug} total")
        
        print(f"Total: {total_original} original + {total_augmented} augmented = {total_original + total_augmented} total")
        
        if len(X) == 0:
            raise ValueError("No valid images found! Please run data_collection.py first.")
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y, model_type="ensemble", cv_folds=5):
        """
        Train a machine learning model on the provided data
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train ('knn', 'svm', 'rf', or 'ensemble')
            cv_folds: Number of cross-validation folds
            
        Returns:
            trained_model: The trained model
            label_encoder: Label encoder for converting between string and numeric labels
        """
        print(f"\nTraining {model_type} model with {len(X)} images...")
        
        # Convert labels to numeric
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Simple hyperparameter tuning for KNN (lightweight approach)
        if model_type == "knn":
            print("\nPerforming simple hyperparameter tuning for KNN...")
            param_candidates = [1, 3, 5, 7, 9]
            best_score = 0
            best_k = 1
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            for k in param_candidates:
                knn_test = KNeighborsClassifier(n_neighbors=k)
                knn_test.fit(X_train_pca, y_train)
                score = knn_test.score(X_test_pca, y_test)
                print(f"k={k}, accuracy={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            print(f"Best k found: {best_k} with accuracy={best_score:.4f}")
            
            # Create the final KNN model with the best k
            model = Pipeline([
                ('pca', PCA(n_components=0.95)),
                ('knn', KNeighborsClassifier(n_neighbors=best_k))
            ])
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Final KNN model accuracy: {accuracy:.4f}")
            
            return model, label_encoder
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define the model based on model_type
        if model_type == "knn":
            # KNN Pipeline with PCA (this is now handled above)
            pass
            
        elif model_type == "svm":
            # SVM Pipeline with PCA
            model = Pipeline([
                ('pca', PCA(n_components=0.95)),
                ('svm', SVC(probability=True))
            ])
            
            # Parameter grid for SVM
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto'],
                'svm__kernel': ['rbf', 'linear']
            }
            
        elif model_type == "rf":
            # Random Forest Pipeline with PCA
            model = Pipeline([
                ('pca', PCA(n_components=0.95)),
                ('rf', RandomForestClassifier(random_state=42))
            ])
            
            # Parameter grid for RF
            param_grid = {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5]
            }
            
        elif model_type == "ensemble":
            # Use all three models and pick the best one
            models = {
                'knn': Pipeline([
                    ('pca', PCA(n_components=0.95)),
                    ('knn', KNeighborsClassifier(n_neighbors=5))
                ]),
                'svm': Pipeline([
                    ('pca', PCA(n_components=0.95)),
                    ('svm', SVC(kernel='rbf', probability=True))
                ]),
                'rf': Pipeline([
                    ('pca', PCA(n_components=0.95)),
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
            }
            
            # Train all models
            best_model = None
            best_accuracy = 0
            
            for name, m in models.items():
                print(f"\nTraining {name} model...")
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"{name} accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = m
                    best_model_name = name
            
            print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
            return best_model, label_encoder
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Use grid search to find best parameters
        print(f"\nPerforming grid search with {cv_folds}-fold cross-validation...")
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
        )
        
        # Train the model with grid search
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        # Return the trained model and label encoder
        return best_model, label_encoder
    
    def evaluate_model(self, model, X, y, label_encoder):
        """
        Evaluate the model and produce performance metrics and visualizations
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels (string format)
            label_encoder: Label encoder used during training
            
        Returns:
            metrics: Dictionary of performance metrics
        """
        print("\nEvaluating model performance...")
        
        # Convert labels to numeric
        y_encoded = label_encoder.transform(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(label_encoder.classes_))
        plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, label_encoder.classes_)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the confusion matrix plot
        os.makedirs("screenshots", exist_ok=True)
        plt.savefig("screenshots/confusion_matrix.png")
        print("Confusion matrix saved to screenshots/confusion_matrix.png")
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics
    
    def save_model(self, model, label_encoder, model_name="rps_model"):
        """
        Save the trained model and label encoder
        
        Args:
            model: Trained model
            label_encoder: Label encoder
            model_name: Base name for the saved model
            
        Returns:
            model_path: Path to the saved model
        """
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d%H%M%S")
        model_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}.pkl")
        
        # Save the model and label encoder
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
        
        # Also save a copy as the latest model
        latest_path = os.path.join(self.model_dir, f"{model_name}_latest.pkl")
        with open(latest_path, 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
        
        # Also save with joblib for compatibility
        joblib_path = os.path.join(self.model_dir, f"{model_name}_latest.joblib")
        joblib.dump({'model': model, 'label_encoder': label_encoder}, joblib_path)
        
        print(f"\nModel saved to:")
        print(f"- {model_path}")
        print(f"- {latest_path}")
        print(f"- {joblib_path}")
        
        # Create a copy in the root directory for compatibility with previous code
        with open(f"{model_name}.pkl", 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
        
        return model_path

def main():
    """Main function to train and evaluate the model"""
    try:
        print("=" * 50)
        print("ROCK PAPER SCISSORS - MODEL TRAINING")
        print("=" * 50)
        
        # Initialize the model trainer
        trainer = RockPaperScissorsModel(data_dir="data", model_dir="models")
        
        # Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        X, y = trainer.load_and_preprocess_data(use_augmentation=True, use_cropped=True)
        
        # Train the model
        print("\nStep 2: Training models...")
        print("\nWhich model would you like to train?")
        print("1. KNN (K-Nearest Neighbors)")
        print("2. SVM (Support Vector Machine)")
        print("3. Random Forest")
        print("4. Ensemble (try all models and pick the best)")
        
        choice = input("Enter choice (1-4, default is 4): ").strip()
        
        if choice == '1':
            model_type = "knn"
        elif choice == '2':
            model_type = "svm"
        elif choice == '3':
            model_type = "rf"
        else:
            model_type = "ensemble"
        
        model, label_encoder = trainer.train_model(X, y, model_type=model_type)
        
        # Evaluate the model
        print("\nStep 3: Evaluating the model...")
        metrics = trainer.evaluate_model(model, X, y, label_encoder)
        
        # Save the model
        print("\nStep 4: Saving the model...")
        model_path = trainer.save_model(model, label_encoder)
        
        print("\nModel training completed successfully!")
        print(f"You can now use the model for prediction by running game.py")
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 