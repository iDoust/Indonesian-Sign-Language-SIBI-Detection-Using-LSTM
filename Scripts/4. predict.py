"""
Script for real-time SIBI sign language prediction using a trained LSTM model.
Captures webcam feed, extracts keypoints, and visualizes predictions.
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    prob_viz,
    DEFAULT_MODEL_PATH,
    ACTIONS,
    SEQUENCE_LENGTH,
    mp_holistic
)

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time SIBI detection using LSTM.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, 
                        help=f"Path to the trained .h5 model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=0.75, 
                        help="Confidence threshold for prediction (default: 0.75)")
    parser.add_argument("--actions", nargs="+", default=ACTIONS.tolist(),
                        help="List of action labels (must match model training)")
    parser.add_argument("--webcam", type=int, default=0,
                        help="Webcam index (default: 0)")
    return parser.parse_args()

def run_prediction(args):
    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prediction variables
    sequence = []
    sentence = []
    predictions = []
    
    # Visualization colors
    colors = [
        (245, 221, 173), (245, 185, 265), (146, 235, 193), 
        (204, 152, 295), (255, 217, 179), (0, 0, 179)
    ]
    
    last_update_time = None
    stable_time = 1.0  # seconds required for "stable" prediction

    cap = cv2.VideoCapture(args.webcam)
    # Set higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 517)

    print("\nStarting prediction... Press 'Q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. MediaPipe Detection
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # 2. Keypoint Extraction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            # 3. Prediction Logic
            if len(sequence) == SEQUENCE_LENGTH:
                # Expand dims to (1, 30, 258)
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))
                
                current_action = args.actions[np.argmax(res)]
                current_prob = res[np.argmax(res)]

                # 4. Stability Check for Sentence Building
                if current_prob > args.conf:
                    current_time = time.time()
                    if last_update_time is None:
                        last_update_time = current_time
                    elif current_time - last_update_time >= stable_time:
                        # Only add if it's different from the last word
                        if len(sentence) == 0 or (sentence[-1] != current_action):
                            sentence.append(current_action)
                            last_update_time = current_time
                else:
                    last_update_time = None

                # Keep only last 5 words in sentence
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # 5. Visualization
                image = prob_viz(res, np.array(args.actions), image, colors, args.conf)

            # Draw sentence bar at the bottom
            cv2.rectangle(image, (0, image.shape[0] - 40), (image.shape[1], image.shape[0]), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show frame
            cv2.imshow('SIBI Detection LSTM', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    run_prediction(args)
