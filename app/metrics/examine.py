"""
Analysis Module: Diagnostics and Validation.

This module provides tools for inspecting the integrity of the ETL pipeline's output data and auditing the performance of trained models. It ensures datasets are balanced and normal, and generates precision/recall matrices to evaluate model convergence.
"""
# Standard Libraries
import logging
from pathlib import Path

# External Libraries
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from jinja2 import Environment, FileSystemLoader

# Application Libraries
from app.models.config import GolemConfig
from app.models.dataset import DoomStreamingDataset
from app.models.brain import DoomLiquidNet
from app.utils.conf import resolve_path, register_command
from app.utils.model import apply_latest_parameters


logger = logging.getLogger(__name__)

# Captum expects a standard tensor-in, tensor-out pass for its hooks.
class ModelWrapperStep(torch.nn.Module):
    def __init__(self, core_model, state):
        super().__init__()
        self.model = core_model
        self.hx = state
    def forward(self, xv, xa=None, xt=None):
        logits, _ = self.model(xv, xa, xt, self.hx)
        return logits[:, 0, :] # Extract the 1 step batch
        
        
def get_last_conv(module_seq):
    for layer in reversed(module_seq):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


@register_command("examine")
def examine(cfg: GolemConfig, module_name: str = "all", target_file: str = None, index: int = 0):
    r"""
    Generates a phenomenological saliency map (Grad-CAM) for a specific sequence.
    
    This evaluates the model's visual and thermal cortices to identify which spatial 
    pixels triggered the agent's highest-probability action prediction.
    
    Args:
        cfg (GolemConfig): Centralized configuration object.
        module_name (str, optional): The specific module dataset to pull a sequence from. Default: ``"all"``
        target_file (str, optional): The specific model file to load.
        index (int, optional): The batch index in the dataset to examine. Default: 0
    """
    try:
        import matplotlib.pyplot as plt
        from captum.attr import LayerGradCam, LayerAttribution
    except ImportError:
        logger.error("Captum and matplotlib are required. Run: pip install captum matplotlib")
        return

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")
    active_profile = cfg.brain.mode
    data_dir = Path(resolve_path(cfg.data.dirs["training"])) / active_profile
    
    # 1. Load the Dataset
    print(data_dir)
    if module_name and module_name.lower() == "all":
        file_pattern = f"{cfg.data.prefix}*.npz"
    else:
        file_pattern = f"{cfg.data.prefix}{module_name}*.npz"
        
    print(file_pattern)
    dataset = DoomStreamingDataset(
        [ str(data_dir) ], 
        seq_len=cfg.training.sequence_length,
        file_pattern=file_pattern,
        augment=False,
        action_names=cfg.training.action_names,
        dsp_config=cfg.brain.dsp,
        sensors=cfg.brain.sensors
    )
    
    if len(dataset) == 0:
        logger.error("No data found to examine.")
        return

    safe_index = min(max(0, index), len(dataset) - 1)
    logger.info(f"Loading sequence {safe_index} from dataset...")
    inputs, actions = dataset[safe_index]

    # 2. Discover Brain Architecture & Load Model
    model_dir = Path(resolve_path(cfg.data.dirs["model"])) / active_profile
    
    if target_file:
        active_model_path = model_dir / target_file
        if not active_model_path.exists():
            logger.error(f"Target model file not found: {active_model_path}")
            return
        archives = [active_model_path]
    else:
        active_model_path = Path(resolve_path(cfg.data.dirs["training"])) / active_profile / "golem.pth"
        archives = list(model_dir.glob("*.pth"))
        
    apply_latest_parameters(cfg, archives)
    
    try:
        state_dict = torch.load(str(active_model_path), map_location=device, weights_only=True)
        n_actions = cfg.training.action_space_size
        if 'output.weight' in state_dict:
            n_actions = state_dict['output.weight'].shape[0]
            
        model = DoomLiquidNet(
            n_actions=n_actions,
            cortical_depth=cfg.brain.cortical_depth,
            working_memory=cfg.brain.working_memory,
            sensors=cfg.brain.sensors,
            dsp_config=cfg.brain.dsp
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        logger.error(f"No brain found at {active_model_path}. Train first!")
        return

    # 3. Prepare Tensors
    # Add batch dimension: (1, Seq_Len, C, H, W)
    x_vis = inputs['visual'].unsqueeze(0).to(device)
    x_aud = inputs['audio'].unsqueeze(0).to(device) if 'audio' in inputs else None
    x_thm = inputs['thermal'].unsqueeze(0).to(device) if 'thermal' in inputs else None
    
    seq_len = x_vis.size(1)
    
    # 4. Separate history (to build ODE state) from the final prediction frame
    if seq_len > 1:
        x_vis_hist = x_vis[:, :-1, ...]
        x_aud_hist = x_aud[:, :-1, ...] if x_aud is not None else None
        x_thm_hist = x_thm[:, :-1, ...] if x_thm is not None else None
        
        with torch.no_grad():
            _, hx = model(x_vis_hist, x_aud_hist, x_thm_hist)
    else:
        hx = None
        
    x_vis_step = x_vis[:, -1:, ...]
    x_aud_step = x_aud[:, -1:, ...] if x_aud is not None else None
    x_thm_step = x_thm[:, -1:, ...] if x_thm is not None else None

    # 5. Captum Wrapper
    wrapper = ModelWrapperStep(model, hx)
    
    # Find the most probable action to attribute
    with torch.no_grad():
        final_logits = wrapper(x_vis_step, x_aud_step, x_thm_step)
        probs = torch.sigmoid(final_logits)[0]
    
    target_idx = torch.argmax(probs).item()
    
    # Resolve the name dynamically in case the loaded weights expanded the action space
    action_names = list(cfg.training.action_names)
    if len(action_names) < n_actions:
        action_names += [f"ACTION_{i}" for i in range(len(action_names), n_actions)]
    target_name = action_names[target_idx]
    
    logger.info(f"Generating Grad-CAM attributing to highest predicted action: {target_name} ({probs[target_idx]:.2f})")

    # 6. Extract Visual Cortex Heatmap
    vis_layer = get_last_conv(model.conv)
    lgc_vis = LayerGradCam(wrapper, vis_layer)
    attr_vis = lgc_vis.attribute(x_vis_step, target=target_idx, additional_forward_args=(x_aud_step, x_thm_step))
    attr_vis = LayerAttribution.interpolate(attr_vis, (64, 64))
    
    attr_vis_np = attr_vis.squeeze().cpu().detach().numpy()
    attr_vis_np = np.maximum(attr_vis_np, 0)
    if np.max(attr_vis_np) > 0:
        attr_vis_np /= np.max(attr_vis_np)

    # 7. Extract Thermal Cortex Heatmap (If active)
    attr_thm_np = None
    if model.use_thermal:
        thm_layer = get_last_conv(model.thermal_conv)
        lgc_thm = LayerGradCam(wrapper, thm_layer)
        attr_thm = lgc_thm.attribute(x_vis_step, target=target_idx, additional_forward_args=(x_aud_step, x_thm_step))
        attr_thm = LayerAttribution.interpolate(attr_thm, (64, 64))
        
        attr_thm_np = attr_thm.squeeze().cpu().detach().numpy()
        attr_thm_np = np.maximum(attr_thm_np, 0)
        if np.max(attr_thm_np) > 0:
            attr_thm_np /= np.max(attr_thm_np)

    # 8. Render Side-by-Side Validation
    img_vis_rgb = x_vis_step[0, 0, :3, ...].permute(1, 2, 0).cpu().numpy()
    
    cols = 4 if model.use_thermal else 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 2: axes = [axes[0], axes[1]] # Ensure list formatting

    # Plot Visual
    axes[0].imshow(img_vis_rgb)
    axes[0].set_title("Visual Input (RGB)")
    axes[0].axis('off')
    
    axes[1].imshow(img_vis_rgb)
    axes[1].imshow(attr_vis_np, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Visual Grad-CAM\nTarget: {target_name}")
    axes[1].axis('off')
    
    # Plot Thermal
    if model.use_thermal:
        img_thm = x_thm_step[0, 0, 0, ...].cpu().numpy()
        axes[2].imshow(img_thm, cmap='gray')
        axes[2].set_title("Thermal Input (Mask)")
        axes[2].axis('off')
        
        axes[3].imshow(img_thm, cmap='gray')
        axes[3].imshow(attr_thm_np, cmap='jet', alpha=0.5)
        axes[3].set_title(f"Thermal Grad-CAM\nTarget: {target_name}")
        axes[3].axis('off')

    out_path = Path("examine.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saliency map saved successfully to: {out_path.absolute()}")