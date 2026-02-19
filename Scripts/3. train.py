"""
Script to train an LSTM model for SIBI sign language detection.
Loads .npy data sequences, defines model architecture, and trains with callbacks.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam

from utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    SEQUENCE_LENGTH,
    TOTAL_KEYPOINTS
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model for SIBI detection.")
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, 
                        help=f"Path to the dataset directory (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, 
                        help=f"Path to save the trained model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train (default: 100)")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of the dataset to include in the test split (default: 0.2)")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate for Adam optimizer (default: 0.001)")
    return parser.parse_args()

class CustomStopCallback(Callback):
    """Custom callback as used in the original TrainingLSTM.ipynb"""
    def on_epoch_end(self, epoch, logs={}):
        # Logic: Stop if epoch > 50, loss < 0.09, and accuracy between 0.88 and 0.99
        if epoch > 50 and logs.get('loss') < 0.09 and (0.88 < logs.get('accuracy') < 0.99):
            print("\nThresholds met. Stopping training early.")
            self.model.stop_training = True

def load_data(data_path):
    print(f"Loading data from: {data_path}")
    actions = np.array([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(data_path, action)
        sequence_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        
        print(f"  Loading '{action}' ({len(sequence_dirs)} sequences)...")
        for sequence_dir in sequence_dirs:
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                # Using standard naming format
                file_path = os.path.join(action_path, sequence_dir, f"{action}_keypoints_{frame_num}.npy")
                if not os.path.exists(file_path):
                    # Fallback for inconsistent naming in old datasets if necessary
                    continue
                res = np.load(file_path)
                window.append(res)
            
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return X, y, actions

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(args):
    # Load and preprocess data
    X, y, actions = load_data(args.path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y)
    
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Classes: {actions}")

    # Build and compile model
    model = build_model((SEQUENCE_LENGTH, TOTAL_KEYPOINTS), actions.shape[0])
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.model_path), 'logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    custom_stop = CustomStopCallback()
    
    # Training
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=[tb_callback, custom_stop]
    )

    # Save model
    model.save(args.model_path)
    print(f"\nModel saved to: {args.model_path}")

    # Evaluation
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Evaluation -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
