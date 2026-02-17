import numpy as np
import os
import sys

def inspect(file_path):
    # Resolve absolute path if a relative one is passed
    if not os.path.isabs(file_path):
        # Try finding it relative to CWD first
        if not os.path.exists(file_path):
            # If not found, try finding it relative to this script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, os.path.basename(file_path))

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print(f"Usage: python app/inspect.py [path_to_data.npz]")
        return

    print(f"--- Analyzing: {os.path.basename(file_path)} ---")
    print(f"Full Path: {file_path}")
    
    # Load the compressed file
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Failed to load numpy file: {e}")
        return
    
    # 1. Inspect Keys
    print(f"Arrays found: {list(data.keys())}")
    
    frames = data['frames']
    actions = data['actions']
    
    # 2. Inspect Shapes
    print(f"\n[Frames]")
    print(f"  Shape: {frames.shape} (Count, Height, Width, Channels)")
    print(f"  Type:  {frames.dtype}")
    print(f"  Range: Min {frames.min():.2f} / Max {frames.max():.2f}")
    
    if frames.max() > 1.0:
        print("  WARNING: Frames are not normalized (0-255). Network expects 0-1.")
    else:
        print("  OK: Frames are normalized (0-1).")

    # 3. Inspect Actions (Class Balance)
    print(f"\n[Actions]")
    print(f"  Shape: {actions.shape}")
    
    total_frames = len(actions)
    if total_frames == 0:
        print("Error: No frames recorded.")
        return

    # Calculate totals for each button [Left, Right, Attack]
    total_presses = np.sum(actions, axis=0)
    
    # Handle potentially different action sizes if config changed
    labels = ["Left", "Right", "Attack"]
    for i, label in enumerate(labels):
        if i < len(total_presses):
            count = int(total_presses[i])
            pct = count / total_frames
            print(f"  {label:<10}: {count} ({pct:.1%})")
    
    # Check for "Do Nothing" frames (all zeros)
    non_action_frames = np.sum(~actions.any(axis=1))
    idle_pct = non_action_frames / total_frames
    print(f"  Idling    : {non_action_frames} ({idle_pct:.1%})")

    if idle_pct > 0.5:
        print("\nWARNING: High idle time (>50%). The bot might learn to stand still.")
    elif idle_pct < 0.1:
        print("\nNOTE: Very low idle time. Good for aggressive behavior.")
    else:
        print("\nOK: Balanced activity levels.")

if __name__ == "__main__":
    # If argument provided, use it. 
    # Otherwise, default to looking for 'doom_training_data.npz' in the same folder as this script.
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target = os.path.join(script_dir, "doom_training_data.npz")
        
    inspect(target)