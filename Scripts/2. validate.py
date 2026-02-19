"""
Script to validate collected .npy keypoint data.
Checks if any files contain only zero values (indicating failed MediaPipe detection).
"""

import os
import argparse
import numpy as np
from utils import DEFAULT_DATA_PATH, SEQUENCE_LENGTH

def parse_args():
    parser = argparse.ArgumentParser(description="Validate collected sign language data.")
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, 
                        help=f"Path to the dataset directory (default: {DEFAULT_DATA_PATH})")
    return parser.parse_args()

def validate_data(args):
    if not os.path.exists(args.path):
        print(f"Error: Dataset path '{args.path}' does not exist.")
        return

    print(f"Validating dataset at: {args.path}")
    
    total_files = 0
    zero_files = 0
    invalid_sequences = []

    actions = [d for d in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, d))]
    
    if not actions:
        print("No action folders found in the dataset.")
        return

    for action in actions:
        action_path = os.path.join(args.path, action)
        sequences = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        
        print(f"Checking action: '{action}' ({len(sequences)} sequences)")
        
        for sequence in sequences:
            sequence_path = os.path.join(action_path, sequence)
            sequence_has_zero = False
            
            for i in range(SEQUENCE_LENGTH):
                # Handle potential naming variations
                filename = f"{action}_keypoints_{i}.npy"
                file_path = os.path.join(sequence_path, filename)
                
                if not os.path.exists(file_path):
                    # Try alternative naming if needed, but standardizing on {action}_keypoints_{i}.npy
                    continue
                
                total_files += 1
                try:
                    keypoints = np.load(file_path)
                    if np.all(keypoints == 0):
                        print(f"  [X] Zero values found: {action}/{sequence}/{filename}")
                        zero_files += 1
                        sequence_has_zero = True
                except Exception as e:
                    print(f"  [!] Error loading {file_path}: {e}")
                    zero_files += 1
                    sequence_has_zero = True
            
            if sequence_has_zero:
                invalid_sequences.append(f"{action}/{sequence}")

    print("\n" + "="*30)
    print("VALIDATION SUMMARY")
    print("="*30)
    print(f"Total files checked: {total_files}")
    print(f"Files with errors/zeros: {zero_files}")
    
    if zero_files == 0:
        print("\nSUCCESS: All data files are valid!")
    else:
        print(f"\nWARNING: Found {len(invalid_sequences)} invalid sequences:")
        for seq in invalid_sequences:
            print(f"  - {seq}")
        print("\nConsider re-collecting data for these sequences.")

if __name__ == "__main__":
    args = parse_args()
    validate_data(args)
