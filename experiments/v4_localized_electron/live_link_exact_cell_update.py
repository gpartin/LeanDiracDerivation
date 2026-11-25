"""Free one-cell update of the full-amplitude localized carrier.

The previous and current committed layers describe the literal same-assignment
carrier advancing by one cell per update.  The action then solves the next
layer without a target force, projection, reset, or receiver preload.  The
exact translated next layer is used only as the post-solve comparator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as prior_action
import live_clifford_bond_action as action
import live_clifford_nonlinear_gate as nonlinear
import live_clifford_tangent_gate as tangent


HERE = Path(__file__).resolve().parent
SIDE = 5


def pack(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def translate(raw: np.ndarray, axis: int, amount: int,
              psi_phase: float, chi_phase: float = 0.0) -> np.ndarray:
    tensor = torch.tensor(raw, dtype=torch.float64)
    psi, chi = native_box.unpack(tensor, SIDE)
    psi = torch.roll(psi, amount, dims=axis)*torch.exp(
        torch.tensor(1j*psi_phase, dtype=torch.complex128))
    chi = torch.roll(chi, amount, dims=axis)*torch.exp(
        torch.tensor(1j*chi_phase, dtype=torch.complex128))
    return pack(psi, chi).detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--branch", choices=("native", "conjugate"),
                        default="native")
    args = parser.parse_args()

    bare, _ = prior_action.expected_blocks()
    current = bare["rest"].copy()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    if args.branch == "conjugate":
        psi, chi = native_box.unpack(torch.tensor(current), SIDE)
        current = pack(torch.conj(psi), torch.conj(chi)).detach().numpy()
        theta = -theta
    previous = translate(
        current, args.axis, -args.sign, -theta)
    target = translate(
        current, args.axis, args.sign, theta)

    scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
    tangent.CACHE = (HERE
                     / f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}.npz")
    rotation = native_spectrum.rotation_matrix(current.size, theta)
    rest_layers = (rotation.T@current, current, rotation@current)
    _bp, _bc, bf = tangent.tangent_blocks(rest_layers)
    factor = lu_factor(bf)

    target_residual = nonlinear.residual(previous, current, target)
    solved, history, converged, _factor = nonlinear.solve_following(
        previous, current, target, factor, maximum_iterations=48)
    target_norm = np.linalg.norm(target)
    scale = native_time.B*native_time.CHI0*max(native_time.DT, 1e-30)
    result = {
        "status": "V4_LIVE_LINK_EXACT_CELL_UPDATE",
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "axis": args.axis,
        "sign": args.sign,
        "carrier_branch": args.branch,
        "dt": native_time.DT,
        "one_cell_per_update": True,
        "psi_clock_phase_per_update": theta,
        "target_DEL_residual_RMS": float(
            np.linalg.norm(target_residual)/np.sqrt(target_residual.size)),
        "target_scaled_DEL_residual_RMS": float(
            np.linalg.norm(target_residual)/np.sqrt(target_residual.size)/scale),
        "solver_converged": converged,
        "solver_iterations": len(history)-1,
        "solver_final_residual_RMS": history[-1],
        "solved_vs_exact_translation_relative_error": float(
            np.linalg.norm(solved-target)/max(target_norm, 1e-300)),
        "solved_norm_ratio": float(np.linalg.norm(solved)/target_norm),
        "verdict": (
            "EXACT_ONE_CELL_NATIVE_UPDATE_PASS"
            if converged and np.linalg.norm(solved-target)/target_norm < 1e-8
            else "EXACT_ONE_CELL_NATIVE_UPDATE_OPEN"),
        "claim_boundary": (
            "One unrestricted full-amplitude native update from an exact "
            "moving two-layer preparation. Repeated handoffs, all directions, "
            "Dirac current, refinement and V4-006 remain separate gates."),
    }
    suffix = action.LINK_SOURCE.replace(".", "p")
    output = HERE/f"live_link_exact_cell_update_{suffix}_{args.branch}_a{args.axis}_s{args.sign}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
