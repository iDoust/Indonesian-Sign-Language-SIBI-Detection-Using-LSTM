"""
Script to collect webcam data for LSTM-based sign language detection.
Uses MediaPipe Holistic to extract 258 keypoints and saves them as .npy sequences.
"""

import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
from utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    DEFAULT_DATA_PATH,
    NUM_SEQUENCES,
    SEQUENCE_LENGTH,
    mp_holistic
)

def parse_args():
    parser = argparse.ArgumentParser(description="Collect sign language data via webcam.")
    parser.add_argument("--actions", nargs="+", default=["Hai"], 
                        help="List of actions to collect (e.g., --actions Hai Nama Saya)")
    parser.add_argument("--sequences", type=int, default=NUM_SEQUENCES, 
                        help=f"Number of video sequences per action (default: {NUM_SEQUENCES})")
    parser.add_argument("--length", type=int, default=SEQUENCE_LENGTH, 
                        help=f"Number of frames per sequence (default: {SEQUENCE_LENGTH})")
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, 
                        help=f"Path to save collected data (default: {DEFAULT_DATA_PATH})")
    return parser.parse_args()

def collect_data(args):
    # Create directories
    for action in args.actions:
        action_path = os.path.join(args.path, action)
        os.makedirs(action_path, exist_ok=True)
        
        # Determine starting sequence number (to avoid overwriting)
        existing_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        dirmax = 0
        if existing_dirs:
            dirmax = np.max(np.array(existing_dirs).astype(int))
            
        print(f"Collecting data for action: '{action}'. Starting from sequence: {dirmax + 1}")

    cap = cv2.VideoCapture(0)
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in args.actions:
            # Re-calculating dirmax for each action in case it changed
            action_path = os.path.join(args.path, action)
            existing_dirs = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
            dirmax = np.max(np.array(existing_dirs).astype(int)) if existing_dirs else 0

            for sequence in range(1, args.sequences + 1):
                subdir_num = dirmax + sequence
                subdir_path = os.path.join(action_path, str(subdir_num))
                os.makedirs(subdir_path, exist_ok=True)

                for frame_num in range(args.length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame. Exiting...")
                        break

                    # Process with MediaPipe
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Show collection status
                    status_text = f'Collecting for {action} | Sequence {subdir_num}/{dirmax + args.sequences} | Frame {frame_num}/{args.length}'
                    
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, status_text, (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed - Data Collection', image)
                        cv2.waitKey(2000) # Give user 2 seconds to prepare
                    else:
                        cv2.putText(image, status_text, (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed - Data Collection', image)

                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(subdir_path, f'{action}_keypoints_{frame_num}.npy')
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Collection interrupted by user.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed successfully.")

if __name__ == "__main__":
    args = parse_args()
    collect_data(args)
