"""Build one live-action tangent block in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank
import live_clifford_bond_action as action
import live_clifford_tangent_gate as gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    dt = gate.DT
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    layers = (
        gate.phase_rotation_matrix(rest.size, -theta)@rest,
        rest,
        gate.phase_rotation_matrix(rest.size, theta)@rest,
    )
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in layers)
    selected = args.index

    def function(value):
        trial = list(tensors)
        trial[selected] = value
        return action.del_residual(*trial, dt)

    block = torch.func.jacrev(
        function, chunk_size=args.chunk_size)(tensors[selected]).detach().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, block)
    print(f"built block {selected}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
