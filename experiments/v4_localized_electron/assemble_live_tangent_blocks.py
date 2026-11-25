"""Assemble three isolated tangent blocks into the gate cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--following", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    np.savez_compressed(
        args.output,
        previous=np.load(args.previous),
        current=np.load(args.current),
        following=np.load(args.following),
    )
    print(args.output)


if __name__ == "__main__":
    main()
