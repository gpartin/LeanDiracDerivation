"""Continuum/low-momentum audit for the odd rotor-wake kinetic basis.

The candidate may not alter the quadratic uniform-vacuum symbol from which
the existing free Dirac readout is derived.  This script verifies that the
candidate starts at quartic perturbation order and records convergence of the
C19 spatial symbol to its isotropic continuum form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_transfer_wake_kinetic_gate as wake
import canonical_transfer_oscillator_gate as oscillator
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5
MODE = "transfer_odd"


def c19_symbol(momentum: np.ndarray) -> float:
    return float(sum(weight*(1-np.cos(np.dot(momentum, offset)))
                     for offset, weight in native.DIRECTIONS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", choices=("kinetic", "oscillator"),
                        default="kinetic")
    args = parser.parse_args()
    extra_interval = (wake.extra_interval if args.completion == "kinetic"
                      else oscillator.oscillator_interval)
    psi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
    chi = np.zeros_like(psi)
    chi[..., 2] = small.CHI0
    vacuum = box.pack(psi, chi)
    rng = np.random.default_rng(4006)
    left_direction = rng.normal(size=vacuum.size)
    right_direction = rng.normal(size=vacuum.size)
    # Normalize so epsilon is the relative field perturbation scale.
    left_direction *= np.linalg.norm(vacuum)/np.linalg.norm(left_direction)
    right_direction *= np.linalg.norm(vacuum)/np.linalg.norm(right_direction)
    epsilons = np.asarray((1e-5, 3e-5, 1e-4, 3e-4, 1e-3))
    values = []
    for epsilon in epsilons:
        left = torch.tensor(
            vacuum+epsilon*left_direction, dtype=torch.float64)
        right = torch.tensor(
            vacuum+epsilon*right_direction, dtype=torch.float64)
        values.append(float(extra_interval(left, right, MODE).detach()))
    slope = float(np.polyfit(np.log(epsilons), np.log(values), 1)[0])

    vacuum_t = torch.tensor(vacuum, dtype=torch.float64)
    combined = torch.cat((vacuum_t, vacuum_t)).requires_grad_(True)

    def doubled(raw: torch.Tensor):
        return extra_interval(
            raw[:vacuum.size], raw[vacuum.size:], MODE)

    gradient = torch.autograd.grad(doubled(combined), combined,
                                   create_graph=True)[0]
    tangent = torch.tensor(np.r_[left_direction, right_direction],
                           dtype=torch.float64)
    hessian_vector = torch.autograd.grad(
        torch.dot(gradient, tangent), combined)[0]

    symbol_rows = []
    directions = {
        "axis": np.asarray((1.0, 0.0, 0.0)),
        "face_diagonal": np.asarray((1.0, 1.0, 0.0))/np.sqrt(2),
        "body_diagonal": np.ones(3)/np.sqrt(3),
    }
    scales = (0.4, 0.2, 0.1, 0.05, 0.025)
    for name, direction in directions.items():
        rows = []
        for scale in scales:
            momentum = scale*direction
            symbol = c19_symbol(momentum)
            rows.append({
                "momentum_norm": scale,
                "c19_symbol": symbol,
                "relative_continuum_remainder": float(symbol/(6*scale**2)-1),
            })
        convergence = float(np.polyfit(
            np.log([row["momentum_norm"] for row in rows]),
            np.log([abs(row["relative_continuum_remainder"])
                    for row in rows]), 1)[0])
        symbol_rows.append({"direction": name, "rows": rows,
                            "remainder_power": convergence})

    hessian_norm = float(torch.linalg.vector_norm(hessian_vector).detach())
    passed = abs(slope-4) < 1e-5 and hessian_norm < 1e-9 and all(
        abs(row["remainder_power"]-2) < 0.05 for row in symbol_rows)
    result = {
        "status": "V4_CANONICAL_TRANSFER_WAKE_CONTINUUM_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate_mode": MODE,
        "completion": args.completion,
        "perturbation_epsilons": epsilons.tolist(),
        "candidate_interval_values": values,
        "candidate_perturbation_power": slope,
        "candidate_uniform_vacuum_hessian_vector_norm": hessian_norm,
        "c19_symbol_normalization": "lambda19(k)/(6 |k|^2)",
        "symbol_rows": symbol_rows,
        "verdict": (
            "CANDIDATE_PRESERVES_QUADRATIC_CONTINUUM_DIRAC_SYMBOL"
            if passed else "CANDIDATE_CONTAMINATES_CONTINUUM_GATE"),
        "interpretation": (
            "The wake metric begins at fourth perturbative order about the "
            "uniform vacuum, so it contributes no quadratic mass or transport "
            "symbol. The C19 symbol retains its isotropic |k|^2 limit with "
            "quadratic remainder. Therefore the existing continuum free-Dirac "
            "factorization is unchanged by this candidate at linear order."),
        "claim_boundary": (
            "Uniform-vacuum quadratic and C19 low-momentum audit only. It does "
            "not prove that the nonlinear localized carrier moves according "
            "to Dirac or that the candidate action is admissible."),
    }
    output = HERE / f"canonical_transfer_{args.completion}_continuum_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
