from __future__ import annotations

import argparse
import json

from modern_transformer.config import load_experiment_config
from modern_transformer.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small decoder-only Transformer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="occasional dotted key=value override")
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.overrides)
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
