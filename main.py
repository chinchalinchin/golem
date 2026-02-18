# Standard Libraries

import argparse
import logging

# Application Libraries

from app.config import GolemConfig
from app.utils import setup_logging
from app.record import record_data
from app.inspect import inspect_data
from app.train import train_agent
from app.run import run_agent

logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Golem: DOOM LNN Agent Manager")
    parser.add_argument("function", choices=["record", "inspect", "train", "run"], help="Operation")
    # Added module argument
    parser.add_argument("--module", help="Specific module to record/train (e.g., 'combat')", default="basic")
    parser.add_argument("--file", help="Specific file to inspect", default=None)
    args = parser.parse_args()

    try:
        cfg = GolemConfig.load()
        setup_logging(cfg.app.log_level)
    except Exception as e:
        print(f"CRITICAL: {e}")
        exit(1)

    # Dispatch
    if args.function == "record":
        # Pass the module name to record
        record_data(cfg, args.module)
    elif args.function == "inspect":
        inspect_data(cfg, args.file)
    elif args.function == "train":
        # Pass module (or None if they want to train everything)
        # Note: If user didn't specify --module, args.module defaults to 'basic'.
        # We might want a separate flag for 'all', or just check if they explicitly passed it.
        # For now, let's treat the default as training 'basic'.
        train_agent(cfg, args.module)
    elif args.function == "run":
        run_agent(cfg)

if __name__ == "__main__":
    main()