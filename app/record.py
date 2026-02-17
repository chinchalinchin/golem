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

# Setup paths
script_dir = get_script_dir()
config_path = os.path.join(script_dir, "custom.cfg")
output_path = os.path.join(script_dir, "doom_training_data.npz")

# Setup game
game = DoomGame()

# 1. Load our custom config
game.load_config(config_path)

# 2. Manually point to the built-in WAD
game.set_doom_scenario_path(get_package_resource("basic.wad"))

game.set_screen_format(ScreenFormat.CRCGCB)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)

# DEBUG: Enable console so you can press '~' to check bindings
game.set_console_enabled(True)

# FORCE BINDINGS: We inject these directly into the engine startup args
# This overrides defaults more reliably than the .cfg file
game.add_game_args("+bind a +moveleft")
game.add_game_args("+bind d +moveright")
game.add_game_args("+bind w +attack") 
game.add_game_args("+bind space +attack")

# SPECTATOR mode allows you to play while the script reads the buffer
game.set_mode(Mode.SPECTATOR) 
game.init()

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
        
        # EXTRACT: Get the raw pixels
        frame = state.screen_buffer 
        
        # TRANSFORM: Resize to 64x64
        frame = cv2.resize(frame.transpose(1, 2, 0), (64, 64))
        
        # TRANSFORM: Normalize to 0-1
        frame = frame / 255.0 
        
        # EXTRACT: Capture your button input
        action = game.get_last_action() 
        
        frames.append(frame)
        actions.append(action)

        game.advance_action()

# LOAD: Save to disk
np.savez_compressed(output_path, frames=np.array(frames), actions=np.array(actions))
print(f"Data saved successfully.")

game.close()