"""
Shared utility functions for SIBI Detection Using LSTM project.
Contains MediaPipe Holistic detection, keypoint extraction, and drawing utilities.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ─── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ─── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# Default actions (classes) for the model
ACTIONS = np.array([
    'aku', 'dia', 'hai', 'kamu', 'maaf',
    'nama', 'no_action', 'sehat', 'terima_kasih', 'tolong'
])

# Default data collection parameters
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'Dataset')
NUM_SEQUENCES = 60
SEQUENCE_LENGTH = 30

# Default model path
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'ModelLSTM-10.h5')

# Number of keypoints per landmark type
POSE_LANDMARKS = 33 * 4      # 33 landmarks × (x, y, z, visibility) = 132
LEFT_HAND_LANDMARKS = 21 * 3  # 21 landmarks × (x, y, z) = 63
RIGHT_HAND_LANDMARKS = 21 * 3 # 21 landmarks × (x, y, z) = 63
TOTAL_KEYPOINTS = POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS  # 258


# ─── Functions ──────────────────────────────────────────────────────────────────

def mediapipe_detection(image, model):
    """
    Run MediaPipe Holistic detection on an image.

    Args:
        image: BGR image from OpenCV
        model: MediaPipe Holistic model instance

    Returns:
        tuple: (processed_image, results)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """
    Extract 258 keypoints from MediaPipe Holistic results.

    Extracts:
    - Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
    - Left hand: 21 landmarks × 3 values (x, y, z) = 63
    - Right hand: 21 landmarks × 3 values (x, y, z) = 63
    Total: 258 keypoints

    Args:
        results: MediaPipe Holistic results object

    Returns:
        numpy.ndarray: Flattened array of 258 keypoint values
    """
    pose = np.array(
        [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(POSE_LANDMARKS)

    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(LEFT_HAND_LANDMARKS)

    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(RIGHT_HAND_LANDMARKS)

    return np.concatenate([pose, lh, rh])


def draw_landmarks(image, results):
    """
    Draw basic landmarks on the image.

    Args:
        image: BGR image from OpenCV
        results: MediaPipe Holistic results object
    """
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks with custom colors on the image.

    Args:
        image: BGR image from OpenCV
        results: MediaPipe Holistic results object
    """
    # Pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def prob_viz(res, actions, input_frame, colors, threshold, max_actions=20):
    """
    Visualize prediction probabilities as colored bars on the frame.

    Args:
        res: Prediction result array
        actions: Array of action names
        input_frame: BGR image frame
        colors: List of BGR color tuples
        threshold: Confidence threshold (unused in visualization, kept for compatibility)
        max_actions: Maximum number of actions to display

    Returns:
        numpy.ndarray: Frame with probability bars overlaid
    """
    overlay = input_frame.copy()
    multiple = 30
    alpha = 0.6

    for num, action in enumerate(actions[:max_actions]):
        prob = res[num] if num < len(res) else 0
        color = colors[num % len(colors)]
        cv2.rectangle(overlay, (0, 30 + num * multiple),
                      (int(prob * 300), 60 + num * multiple), color, -1)

    overlay = cv2.addWeighted(overlay, alpha, input_frame, 1 - alpha, 0)

    for num, action in enumerate(actions[:max_actions]):
        prob = res[num] if num < len(res) else 0
        cv2.putText(overlay, action + ' ' + str(round(prob * 100, 2)) + '%',
                    (5, 50 + num * multiple),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay
