# Standard Libraries
import argparse
import logging
import inspect as py_inspect # Aliased to prevent shadowing the inspect command

# Application Libraries
from app.models.config import GolemConfig
from app.utils import setup_logging, COMMAND_REGISTRY

# Importing these registers the decorated functions in COMMAND_REGISTRY
from app.pipeline import audit, inspect, intervene, train, record, run, summary
from app.client import remote, server, spectate, client

logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Golem: DOOM LNN Agent Manager")
    
    # Dynamically populate choices from the registry
    parser.add_argument("function", choices=list(COMMAND_REGISTRY.keys()), help="Operation to perform")
    parser.add_argument("--module", help="Specific module or 'all'", default="basic")
    parser.add_argument("--file", help="Specific file to inspect", default=None)
    parser.add_argument("--players", type=int, help="Number of players for the host arena", default=3)
    parser.add_argument("--mode", choices=["basic", "classic", "fluid"], help="Override the config brain mode at runtime", default=None)
    parser.add_argument("--full", action="store_true", help="Run a full audit instead of capping at 50 batches")

    args = parser.parse_args()

    try:
        cfg = GolemConfig.load()
        
        # Apply runtime overrides
        if args.mode:
            cfg.brain.mode = args.mode
            
        setup_logging(cfg.app.log_level)
    except Exception as e:
        print(f"CRITICAL: {e}")
        exit(1)

    # Dispatch Pipeline via Registry
    func = COMMAND_REGISTRY.get(args.function)
    if not func:
        logger.error(f"Command '{args.function}' is not registered.")
        exit(1)
        
    # Intelligently map CLI arguments to the target function's signature
    sig = py_inspect.signature(func)
    kwargs = {}
    
    if 'module_name' in sig.parameters:
        kwargs['module_name'] = args.module
    if 'target_file' in sig.parameters:
        kwargs['target_file'] = args.file
    if 'players' in sig.parameters:
        kwargs['players'] = args.players
    if 'full' in sig.parameters:
        kwargs['full'] = args.full

    # Execute the resolved function
    func(cfg, **kwargs)

if __name__ == "__main__":
    main()