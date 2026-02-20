import os
import re
import shutil
from pathlib import Path

def migrate():
    base_dir = Path("data")
    if not base_dir.exists():
        print("Error: Run this script from the project root (/Users/gmoore/Home/lib/golem).")
        return

    # 1. Migrate Training Data (.npz)
    print("--- Migrating Training Data ---")
    # Captures: 1: Profile, 2: Module (including '_recovery' if present), 3: Old Increment
    npz_pattern = re.compile(r"doom_training_(basic|classic|fluid)_(.+)_(\d+)\.npz")
    
    for file_path in base_dir.glob("*.npz"):
        match = npz_pattern.match(file_path.name)
        if match:
            profile = match.group(1)
            module_str = match.group(2)
            
            target_dir = base_dir / profile
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Auto-increment to prevent colliding with existing data
            counter = 1
            while True:
                new_name = f"golem_{module_str}_{counter}.npz"
                target_path = target_dir / new_name
                if not target_path.exists():
                    break
                counter += 1
            
            print(f"Moving {file_path.name} -> {profile}/{new_name}")
            shutil.move(str(file_path), str(target_path))

    # 2. Migrate Models (.pth)
    print("\n--- Migrating Models ---")
    old_models_dir = base_dir / "models"
    new_models_dir = base_dir / "model"
    
    # Captures: 1: Date, 2: Cortical Depth, 3: Working Memory, 4: Profile
    pth_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\.c-(\d+)\.w-(\d+)\.(basic|classic|fluid)\.pth")
    
    if old_models_dir.exists():
        for file_path in old_models_dir.glob("*.pth"):
            match = pth_pattern.match(file_path.name)
            if match:
                date_str = match.group(1)
                c_depth = match.group(2)
                w_mem = match.group(3)
                profile = match.group(4)
                
                target_dir = new_models_dir / profile
                target_dir.mkdir(parents=True, exist_ok=True)
                
                counter = 1
                while True:
                    new_name = f"{date_str}.c-{c_depth}.w-{w_mem}.{counter}.pth"
                    target_path = target_dir / new_name
                    if not target_path.exists():
                        break
                    counter += 1
                    
                print(f"Moving {file_path.name} -> model/{profile}/{new_name}")
                shutil.move(str(file_path), str(target_path))
        
        # Cleanup
        try:
            old_models_dir.rmdir()
            print("Cleaned up old 'data/models/' directory.")
        except OSError:
            print("Old 'data/models/' directory is not empty, skipping removal.")

if __name__ == "__main__":
    migrate()