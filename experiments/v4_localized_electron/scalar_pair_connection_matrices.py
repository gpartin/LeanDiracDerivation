"""Derive complete DEL matrices for each scalar-oriented pair connection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import density_pair_connection_basis_screen as basis
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent


def derive(rest: np.ndarray, feature: str):
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]

    def residual(left, middle, right):
        action = (basis.one_form(left, middle, feature)
                  + basis.one_form(middle, right, feature))
        return torch.autograd.grad(action, middle, create_graph=True)[0]

    matrices = []
    for argument in range(3):
        def function(value, selected=argument):
            trial = layers.copy()
            trial[selected] = value
            return residual(*trial)
        matrices.append(torch.autograd.functional.jacobian(
            function, layers[argument], vectorize=True).detach().numpy())
    minus = matrices[0]@rotation.T
    zero = matrices[1]
    plus = matrices[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", choices=basis.FEATURES)
    args = parser.parse_args()
    bare = np.load(canonical.CACHE)
    blocks = derive(bare["rest"], args.feature)
    path = HERE/f"scalar_pair_connection_{args.feature}_matrices.npz"
    np.savez_compressed(path, **blocks)
    result = {
        "status": "V4_SCALAR_PAIR_CONNECTION_MATRIX_DERIVATION",
        "feature": args.feature,
        "path": path.name,
        "mass_symmetry_residual": float(np.linalg.norm(
            blocks["mass"]-blocks["mass"].T)),
        "gyro_antisymmetry_residual": float(np.linalg.norm(
            blocks["gyro"]+blocks["gyro"].T)),
        "stiffness_symmetry_residual": float(np.linalg.norm(
            blocks["stiffness"]-blocks["stiffness"].T)),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
