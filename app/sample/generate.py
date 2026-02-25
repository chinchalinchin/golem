"""
Procedural Generation Module: Oblige Wrapper.
Generates randomized DOOM maps to cure spatial overfitting and covariate shift.
"""
# Standard Libraries
import logging

# Application Libraries
from app.models.config import GolemConfig
from app.utils.conf import register_command
from app.sample.interfaces import ObligeGenerator

logger = logging.getLogger(__name__)

@register_command("generate")
def generate(cfg: GolemConfig, target_file: str = None):
    """
    Generates a random procedural map and immediately launches a recording session.
    """
    if target_file is None:
        target_file = "temp.wad"

    generator = ObligeGenerator(cfg.randomizer)
    generator.build_map(target_file)