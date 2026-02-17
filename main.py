import argparse
import logging
from app.config import GolemConfig
from app.utils import setup_logging
from app.record import record_data
from app.inspect import inspect_data

logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Golem: DOOM LNN Agent Manager")
    parser.add_argument("function", choices=["record", "inspect"], help="Operation to perform")
    parser.add_argument("--file", help="Specific file to inspect (optional)", default=None)
    args = parser.parse_args()

    # 1. Bootstrap Configuration
    try:
        cfg = GolemConfig.load()
        setup_logging(cfg.app.log_level)
        logger.info(f"Booting {cfg.app.name} v{cfg.app.version}")
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration: {e}")
        exit(1)

    # 2. Dispatch
    if args.function == "record":
        record_data(cfg)
    elif args.function == "inspect":
        inspect_data(cfg, args.file)

if __name__ == "__main__":
    main()