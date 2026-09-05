from __future__ import annotations

import argparse
import json

from modern_transformer.data import prepare_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic token arrays")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_from_config(args.config), indent=2))


if __name__ == "__main__":
    main()
