"""
Multiplayer Host Module.
Initializes a central ViZDoom Arena.
"""
# Standard Libraries
import logging

# External Libraries
from vizdoom import DoomGame, Mode

# Application Libraries
from app.models.config import GolemConfig
from app.utils.conf import resolve_path, register_command
from app.utils.doom import get_scenario


logger = logging.getLogger(__name__)


@register_command("server")
def server(cfg: GolemConfig, module_name: str = "cig_arena", players: int = 3, timelimit: int = 10):
    logger.info(f"Starting Host Arena Server for {players} players...")
    
    game = DoomGame()
    
    active_profile = cfg.brain.mode
    cfg_path = resolve_path(cfg.config[active_profile])
    game.load_config(cfg_path)
    
    if module_name not in cfg.modules:
        logger.error(f"Module '{module_name}' not found. Defaulting to 'deathmatch'.")
        module_name = "deathmatch"
        
    scenario_path = get_scenario(cfg.modules[module_name].scenario)
    game.set_doom_scenario_path(scenario_path)
    
    # +sv_forcerespawn 1: Auto-respawn upon death
    # +sv_spawnfarthest 1: Spawn far from enemies
    # +sv_respawnprotect 1: Brief invulnerability to prevent spawn camping
    host_args = (
        f"-host {players} "
        f"-deathmatch "
        f"+timelimit {timelimit} "
        f"+sv_forcerespawn 1 "
        f"+sv_noautoaim 1 "
        f"+sv_respawnprotect 1 "
        f"+sv_spawnfarthest 1 "
        f"+sv_nocrouch 1 "
        f"+map map01"
    )
    
    game.add_game_args(host_args)
    game.add_game_args("+name ArenaHost +colorset 0")
    
    # Headless Rendering
    game.set_window_visible(False)
    
    # PLAYER mode ensures the host waits for the clients' sync ticks
    game.set_mode(Mode.PLAYER)
    
    game.init()
    logger.info(f"Host Server Initialized on map01. Waiting for {players - 1} clients to join...")
    
    # The host must push empty actions to advance the synchronized logic tick
    empty_action = [0] * cfg.training.action_space_size
    
    episodes = 5
    for i in range(episodes):
        logger.info(f"--- Episode {i+1} Starting ---")
        while not game.is_episode_finished():
            game.make_action(empty_action)
            
            if game.get_episode_time() % 350 == 0:
                logger.info(f"Host Tick: {game.get_episode_time()}")
                
        logger.info(f"--- Episode {i+1} Finished ---")
        game.new_episode()

    game.close()
    logger.info("Host Server Shutdown.")