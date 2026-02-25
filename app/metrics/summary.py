
# Standard Libraries
import logging
from pathlib import Path

# External Libraries
import torch

# Application Libraries
from app.models.config import GolemConfig
from app.models.brain import DoomLiquidNet
from app.utils.conf import resolve_path, register_command
from app.utils.model import apply_latest_parameters


logger = logging.getLogger(__name__)


@register_command("list")
def models(cfg: GolemConfig, mode: str = None):
    r"""
    Lists available model archives.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        mode (str, optional): The specific mode to list models for (e.g., "basic", "fluid"). 
            If ``None``, it evaluates against all available modes. Default: ``None``.
    """
    base_model_dir = Path(resolve_path(cfg.data.dirs["model"]))
    
    modes_to_check = [mode] if mode else list(cfg.config.keys())
    
    logger.info("======================================================")
    logger.info("Available Golem Models")
    logger.info("======================================================")
    
    found_any = False
    for m in modes_to_check:
        m_dir = base_model_dir / m
        if m_dir.exists() and m_dir.is_dir():
            models = list(m_dir.glob("*.pth"))
            if models:
                found_any = True
                logger.info(f"Mode: {m.upper()}")
                for mod in sorted(models, key=lambda f: f.stat().st_mtime, reverse=True):
                    # Calculate size
                    size_mb = mod.stat().st_size / (1024 * 1024)
                    logger.info(f"  - {mod.name} ({size_mb:.2f} MB)")
    
    if not found_any:
        logger.info("No models found.")
    logger.info("======================================================")



@register_command("summary")
def summary(cfg: GolemConfig, module_name: str = None):
    r"""
    Prints a detailed architectural summary of the active brain configuration.

    This function instantiates the LNN based on the current configuration and uses the `torchinfo` package to perform a dummy forward pass. It displays the exact tensor dimensions at each layer, the parameter counts, and validates that the multi-modal sensor fusion layers are properly scaling and concatenating into the Liquid Core.

    Args:
        cfg (GolemConfig): The centralized application configuration object.
        module_name (str, optional): Ignored. Included for CLI compatibility.
    """
    try:
        import torchinfo
    except ImportError:
        logger.error("torchinfo is required for the summary command. Run: pip install torchinfo")
        return

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 1. Base defaults
    active_profile = cfg.brain.mode
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
    n_actions = cfg.training.action_space_size 

    # 2. Discover architecture from archives
    archives = list(model_dir.glob("*.pth"))
    apply_latest_parameters(cfg, archives)
        
    # 3. Discover action space and load state dict (if it exists)
    if active_model_path.exists():
        try:
            state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
            if 'output.weight' in state_dict:
                n_actions = state_dict['output.weight'].shape[0]
        except Exception as e:
            logger.warning(f"Could not load state dict from {active_model_path}: {e}")

    model = DoomLiquidNet(
        n_actions=n_actions,
        cortical_depth=cfg.brain.cortical_depth,
        working_memory=cfg.brain.working_memory,
        sensors=cfg.brain.sensors,
        dsp_config=cfg.brain.dsp
    ).to(device)

    # 4. Construct Multi-Modal Dummy Tensors
    seq_len = cfg.training.sequence_length
    batch_size = 1 
    
    c_vis = 4 if cfg.brain.sensors.depth else 3
    x_vis = torch.randn(batch_size, seq_len, c_vis, 64, 64).to(device)
    
    x_aud = None
    if cfg.brain.sensors.audio:
        # Calculate raw audio samples per frame (44100 Hz / 35 FPS = 1260)
        audio_samples_per_frame = int(cfg.brain.dsp.sample_rate / 35)
        # Dummy tensor now represents raw waveforms, not spectrograms
        x_aud = torch.randn(batch_size, seq_len, 2, audio_samples_per_frame).to(device)
        
    x_thm = None
    if cfg.brain.sensors.thermal:
        x_thm = torch.randn(batch_size, seq_len, 1, 64, 64).to(device)
        
    # Strip None values to prevent torchinfo memory calculation crashes
    input_dict = {"x_vis": x_vis}
    if x_aud is not None:
        input_dict["x_aud"] = x_aud
    if x_thm is not None:
        input_dict["x_thm"] = x_thm

    logger.info("======================================================")
    logger.info(f"Generating Architectural Summary for Profile: {active_profile.upper()}")
    logger.info(f"Sensors Enabled -> Visual: True | Depth: {cfg.brain.sensors.depth} | Audio: {cfg.brain.sensors.audio} | Thermal: {cfg.brain.sensors.thermal}")
    logger.info("======================================================")
    
    torchinfo.summary(
        model, 
        input_data=input_dict, 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3
    )

