# Standard Libraries
import argparse
import logging

# Application Libraries

from app.models.config import GolemConfig
from app.utils import setup_logging
from app.pipeline import audit, inspect, intervene, train, record, run
from app.client import remote, server, spectate, client

logger = logging.getLogger("main")

FUNCTIONS = [
    "record",
    "inspect",
    "train",
    "run",
    "audit",
    "intervene",
    "spectate"
]
def main():
    parser = argparse.ArgumentParser(description="Golem: DOOM LNN Agent Manager")
    parser.add_argument("function", choices=["record", "inspect", "train", "run", "audit", "intervene"], help="Operation")
    parser.add_argument("--module", help="Specific module or 'all'", default="basic")
    parser.add_argument("--file", help="Specific file to inspect", default=None)
    args = parser.parse_args()

    try:
        cfg = GolemConfig.load()
        setup_logging(cfg.app.log_level)
    except Exception as e:
        print(f"CRITICAL: {e}")
        exit(1)

    # Dispatch Pipeline
    if args.function == "record":
        record(cfg, args.module)
    elif args.function == "inspect":
        inspect(cfg, args.file)
    elif args.function == "train":
        train(cfg, args.module)
    elif args.function == "run":
        run(cfg, args.module)
    elif args.function == "audit":
        audit(cfg, args.module)
    elif args.function == "intervene":
        intervene(cfg, args.module)

    # Dispatch Client/Server
    elif args.function == "server":
        server(cfg, module_name=args.module, players=args.players)
    elif args.function == "client":
        client(cfg, module_name=args.module)
    elif args.function == "spectate":
        spectate(cfg, module_name=args.module)
    elif args.function == "remote":
        remote(cfg, module_name=args.module)

if __name__ == "__main__":
    main()