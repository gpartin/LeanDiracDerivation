"""Fixed-orbit tangent screen for a full joint-field projective bond energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
import torch

import action_response_rank as rank
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def unpack(raw: torch.Tensor, side: int):
    data = raw.reshape(side, side, side, 4, 3)
    psi = torch.complex(data[..., 0, :], data[..., 1, :])
    chi = torch.complex(data[..., 2, :], data[..., 3, :])
    return psi, chi


def d19(value: torch.Tensor):
    result = torch.zeros_like(value)
    for offset, weight in field.prior.bond.local.common.base.DIRECTIONS:
        result = result + weight * (
            value - torch.roll(value, shifts=offset, dims=(0, 1, 2)))
    return result


def negative_bond_energy(raw: torch.Tensor, side: int):
    psi, chi = unpack(raw, side)
    joint = torch.cat((psi, chi), dim=-1)
    relation = joint[..., :, None] * joint.conj()[..., None, :]
    density = torch.sum(abs(relation)**2, dim=(-2, -1), keepdim=True)
    normalized = (field.prior.R * relation
                  / torch.sqrt(field.prior.R**4 + density))
    energy = torch.real(torch.sum(normalized.conj() * d19(normalized)))
    return -energy / (2 * field.prior.bond.C19_WEIGHT_SUM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--etas", nargs="+", type=float,
                        default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--output", default="joint_projector_tangent.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = rank.exact_rest(args.rate)
    q = torch.tensor(layers[1], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.functional.jacobian(
        lambda raw: negative_bond_energy(raw, side), q,
        vectorize=True).detach().numpy()
    hessian_l = torch.autograd.functional.hessian(
        lambda raw: negative_bond_energy(raw, side), q,
        vectorize=True).detach().numpy()
    dstiffness = -(hessian_l + hessian_l.T) / 2
    size = mass.shape[0]
    rows = []
    for eta in [0.0] + args.etas:
        trial_stiffness = stiffness + eta * dstiffness
        trial_generator = np.block([
            [np.zeros_like(mass), np.eye(size)],
            [-np.linalg.solve(mass, trial_stiffness),
             -np.linalg.solve(mass, gyro)],
        ])
        values = eig(trial_generator, right=False, check_finite=False)
        rows.append({
            "eta": eta,
            "maximum_growth": float(np.max(values.real)),
            "growth_count_above_1e_2": int(np.sum(values.real > 1e-2)),
            "growth_count_above_1e_4": int(np.sum(values.real > 1e-4)),
        })
    result = {
        "status": "V4_JOINT_PROJECTOR_TANGENT_RESEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rest_del_rms_residual": residual,
        "unit_action_rest_gradient_norm": float(np.linalg.norm(gradient)),
        "stiffness_asymmetry": float(np.linalg.norm(
            dstiffness - dstiffness.T)),
        "rows": rows,
        "claim_boundary": (
            "Fixed-orbit quadratic screen only. A favorable row requires "
            "analytic reciprocal variation and a freely re-solved orbit."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
