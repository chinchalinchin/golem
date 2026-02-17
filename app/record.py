from vizdoom import *
import numpy as np
import cv2
import os

# HELPER: Get absolute path to the directory containing this script
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

# HELPER: Find the built-in scenarios from the python package
def get_package_resource(name):
    package_path = os.path.dirname(vizdoom.__file__)
    return os.path.join(package_path, "scenarios", name)

# HELPER: Generate a unique filename (e.g., data_1.npz, data_2.npz)
def get_unique_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{name}_{counter}{ext}"):
        counter += 1
    return f"{name}_{counter}{ext}"

# Setup paths
script_dir = get_script_dir()
config_path = os.path.join(script_dir, "custom.cfg")
base_output_path = os.path.join(script_dir, "doom_training_data.npz")
output_path = get_unique_filename(base_output_path)

# Setup game
game = DoomGame()

# 1. Load configuration
game.load_config(config_path)
game.set_doom_scenario_path(get_package_resource("basic.wad"))
game.set_screen_format(ScreenFormat.CRCGCB)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR) 
game.init()

# 2. FORCE BINDINGS (The Fix)
# We send these as direct console commands AFTER init.
# This bypasses the command-line parser that was stripping the '+' signs.
print("Applying control bindings...")
game.send_game_command("bind a +moveleft")
game.send_game_command("bind d +moveright")
game.send_game_command("bind w +attack") 
game.send_game_command("bind space +attack")

frames = []
actions = []

print(f"Recording to: {output_path}")
print("Controls: A (Left), D (Right), Space/W (Shoot)")
print("Start playing! Press 'Q' to quit early.")

episodes = 5
for i in range(episodes):
    print(f"Episode {i+1}/{episodes}")
    game.new_episode()
    
    while not game.is_episode_finished():
        state = game.get_state()
        
        # EXTRACT & TRANSFORM
        frame = state.screen_buffer 
        frame = cv2.resize(frame.transpose(1, 2, 0), (64, 64))
        frame = frame / 255.0 
        
        # EXTRACT ACTION
        # returns [MOVE_LEFT, MOVE_RIGHT, ATTACK] as floats, e.g., [1.0, 0.0, 0.0]
        action = game.get_last_action() 
        
        # DEBUG: Print action if any button is pressed
        if any(x > 0 for x in action):
            print(f"Input Detected: {action}")

        frames.append(frame)
        actions.append(action)

        game.advance_action()

# LOAD: Save to disk
np.savez_compressed(output_path, frames=np.array(frames), actions=np.array(actions))
print(f"Data saved successfully to {output_path}")

game.close()