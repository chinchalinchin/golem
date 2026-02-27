import logging
import random
from typing import Dict, Any

from app.models.config import RandomizerConfig
from app.sample.interfaces import ObligeGenerator

logger = logging.getLogger(__name__)

class CurriculumObligeGenerator(ObligeGenerator):
    """
    An extension of the ObligeGenerator that replaces pure uniform random sampling with Stratified Curriculum Learning and Conditional Priors.
    """
    def __init__(self, cfg: RandomizerConfig, phase: int = 1):
        super().__init__(cfg)
        self.phase = phase
        logger.info(f"Initialized Curriculum Generator at Phase {self.phase}")

    def _get_phase_constraints(self) -> Dict[str, list]:
        """
        Defines the parameter space based on the active curriculum phase.
        Phase 1: Simple navigation, minimal temporal complexity.
        Phase 2: Introduction of verticality, traps, and moderate combat.
        Phase 3: The full generalized distribution (complex topology, swarms).
        """
        if self.phase == 1:
            return {
                "size": ["micro", "small"],
                "theme": ["original", "tech"],
                "outdoors": ["none"],
                "steepness": ["none"],
                "liquids": ["none"],
                "teleporters": ["none"],
                "mons": ["none", "sparse"],
                "strength": ["easier", "normal"]
            }
        elif self.phase == 2:
            return {
                "size": ["regular"],
                "theme": ["urban", "hell", "mixed"],
                "outdoors": ["mixed"],
                "steepness": ["mixed"],
                "liquids": ["mixed"],
                "teleporters": ["none"], # Still constrain teleportation to prevent state amnesia
                "mons": ["normal", "lots"],
                "strength": ["normal"]
            }
        else:
            # Phase 3+: Full unbounded distribution defined in app.yaml
            return self.base_oblige_config

    def _apply_conditional_priors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforces topological coherence by mathematically coupling dependent variables.
        Prevents degenerate combinations that generate conflicting gradients.
        """
        # Constraint 1: Volume vs. Density
        # A micro map cannot physically support a swarm without spawning enemies inside each other.
        if params.get("size") == "micro" and params.get("mons") in ["lots", "swarms"]:
            params["mons"] = "normal"
            logger.debug("Prior Triggered: Reduced monster density for micro-sized geometry.")

        # Constraint 2: Verticality vs. Outdoors
        # 'Epic' steepness in tight indoor corridors breaks ViZDoom's pathing heuristics.
        if params.get("steepness") == "plenty" and params.get("outdoors") == "none":
            params["outdoors"] = "mixed"
            logger.debug("Prior Triggered: Forced outdoor regions to accommodate extreme verticality.")

        # Constraint 3: The Teleporter Amnesia Trap
        # Teleporters instantly shift the visual manifold. If the network is untrained (Phase < 3),
        # this causes the CfC memory state to collapse. Ensure teleporters only exist in large maps.
        if params.get("teleporters") == "plenty" and params.get("size") in ["micro", "small"]:
            params["teleporters"] = "none"

        return params

    def sample_configuration(self) -> Dict[str, Any]:
        """
        Generates a valid, constrained random sample based on the current curriculum.
        """
        phase_space = self._get_phase_constraints()
        sampled_params = {}

        # 1. Stratified Sample within the Phase bounds
        for key, base_values in self.base_oblige_config.items():
            # If the phase defines a tighter bound, use it; otherwise, use the base config
            available_options = phase_space.get(key, base_values)
            
            # Fallback to base values if the phase constraint is accidentally empty
            if not available_options: 
                available_options = base_values

            sampled_params[key] = random.choice(available_options) if isinstance(available_options, list) else available_options

        # 2. Apply Bayesian constraints to the sampled vector
        return self._apply_conditional_priors(sampled_params)

    def build_map(self, filename: str = "golem_procgen.wad", overrides: dict = None) -> str:
        """
        Overrides the base build_map to inject the curriculum-sampled parameters 
        before compiling the BSP via the container.
        """
        # Generate the structured sample
        structured_params = self.sample_configuration()
        
        # Allow explicit runtime overrides (e.g., from the CLI) to take final precedence
        if overrides:
            structured_params.update(overrides)

        # Pass the pre-computed dictionary to the parent class, acting as complete overrides
        return super().build_map(filename, overrides=structured_params)