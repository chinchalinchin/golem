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
from app.audit import audit_agent

logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Golem: DOOM LNN Agent Manager")
    parser.add_argument("function", choices=["record", "inspect", "train", "run", "audit"], help="Operation")
    parser.add_argument("--module", help="Specific module or 'all'", default="basic")
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
        record_data(cfg, args.module)
    elif args.function == "inspect":
        inspect_data(cfg, args.file)
    elif args.function == "train":
        train_agent(cfg, args.module)
    elif args.function == "run":
        run_agent(cfg)
    elif args.function == "audit":
        audit_agent(cfg, args.module)

if __name__ == "__main__":
    main()