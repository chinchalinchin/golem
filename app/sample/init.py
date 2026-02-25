"""
Initialization Module: Environment Setup.
"""
# Standard Libraries
import logging
import subprocess
import sys

# Application Libraries
from app.models.config import GolemConfig
from app.utils.conf import register_command

logger = logging.getLogger(__name__)

@register_command("init")
def init(cfg: GolemConfig):
    """
    Initializes the Golem environment by verifying and building required Docker images.
    """
    logger.info("Initializing Golem environment...")
    image_name = "golem-oblige:latest"

    # 1. Check if Docker is installed (executable exists in PATH)
    try:
        subprocess.run(
            ["docker", "--version"], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH. Please install Docker to continue.")
        sys.exit(1)

    # 2. Check if Docker daemon is running
    daemon_check = subprocess.run(
        ["docker", "info"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    if daemon_check.returncode != 0:
        logger.error("Docker daemon is not responsive. Please start Docker and try again.")
        sys.exit(1)

    # 3. Check if the Oblige image already exists
    image_check = subprocess.run(
        ["docker", "image", "inspect", image_name], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

    if image_check.returncode == 0:
        logger.info(f"Docker image '{image_name}' already exists. Ready for procedural generation.")
        return

    # 4. Build the image
    logger.info(f"Docker image '{image_name}' not found. Initiating build...")
    build_cmd = [
        "docker", "buildx", "build",
        "-f", "Dockerfile.oblige",
        "-t", image_name,
        "--load", # Ensures the image is loaded into the local docker daemon
        "."
    ]

    try:
        # Note: We do not capture stdout/stderr here so the user can see 
        # the standard Docker build progression in their terminal.
        subprocess.run(build_cmd, check=True)
        logger.info(f"Successfully built '{image_name}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build '{image_name}'. Process exited with code: {e.returncode}")
        sys.exit(1)