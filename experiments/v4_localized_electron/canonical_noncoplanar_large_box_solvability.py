"""Lyapunov--Schmidt solvability of the unique side-five carrier direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large


HERE = Path(__file__).resolve().parent
FRACTIONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def main() -> None:
    data = np.load(HERE / "canonical_noncoplanar_large_box_audit.npz")
    rest = data["rest"]
    direction = data["direction"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (box.interval(box.phase_rotate(
            raw, np.conjugate(phase), large.SIDE), raw, large.SIDE)
            + box.interval(raw, box.phase_rotate(
                raw, phase, large.SIDE), large.SIDE))

    def residual(raw: np.ndarray) -> np.ndarray:
        value = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(
            orbit_action(value), value)[0].detach().numpy()

    tensor = torch.tensor(rest, dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        orbit_action, tensor, vectorize=True).detach().numpy()
    hessian = (hessian+hessian.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    null_mask = abs(eigenvalues) < 1e-8*scale
    null_basis = eigenvectors[:, null_mask]
    range_basis = eigenvectors[:, ~null_mask]
    range_values = eigenvalues[~null_mask]
    rest_norm = np.linalg.norm(rest)
    rows = []
    states = []
    for fraction in FRACTIONS:
        epsilon = fraction*rest_norm
        state = rest+epsilon*direction
        trace = []
        for iteration in range(12):
            value = residual(state)
            range_coordinates = range_basis.T@value
            null_coordinates = null_basis.T@value
            trace.append({
                "iteration": iteration,
                "total_rms": float(np.linalg.norm(value)
                                   / np.sqrt(value.size)),
                "range_norm": float(np.linalg.norm(range_coordinates)),
                "null_norm": float(np.linalg.norm(null_coordinates)),
            })
            old = np.linalg.norm(range_coordinates)
            if old < 1e-12:
                break
            correction = -range_basis@(range_coordinates/range_values)
            for power in range(12):
                trial = state+0.5**power*correction
                if np.linalg.norm(range_basis.T@residual(trial)) < old:
                    state = trial
                    break
            else:
                break
        final = residual(state)
        null_final = null_basis.T@final
        range_final = range_basis.T@final
        displacement = state-rest
        amplitude = float(direction@displacement)
        psi, chi = large.unpack_np(state)
        psi0, _chi0 = large.unpack_np(rest)
        core = np.sum(abs(psi0)**2, axis=-1)>1e-12
        rows.append({
            "fraction_of_rest_norm": fraction,
            "requested_amplitude": epsilon,
            "retained_direction_amplitude": amplitude,
            "range_correction_norm": float(np.linalg.norm(
                displacement-amplitude*direction)),
            "final_total_rms": float(
                np.linalg.norm(final)/np.sqrt(final.size)),
            "final_range_norm": float(np.linalg.norm(range_final)),
            "final_null_solvability_norm": float(np.linalg.norm(null_final)),
            "physical_direction_projection": float(direction@final),
            "null_norm_over_amplitude_cubed": float(
                np.linalg.norm(null_final)/max(abs(amplitude)**3, 1e-300)),
            "halo_psi_power_fraction": float(
                np.sum(abs(psi[~core])**2)/max(np.sum(abs(psi)**2), 1e-300)),
            "halo_chi_deviation_norm": float(np.linalg.norm(
                chi[~core]-np.asarray((0, 0, small.CHI0)))),
            "trace": trace,
        })
        states.append(state)
    obstructed = all(row["final_null_solvability_norm"] > 1e-10
                     for row in rows[-3:])
    solved = any(row["final_total_rms"] < 1e-10
                 and row["fraction_of_rest_norm"] >= 1e-3 for row in rows)
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_LARGE_BOX_SOLVABILITY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "full_nullity": int(null_basis.shape[1]),
        "rows": rows,
        "verdict": (
            "LARGE_BOX_NONCOPLANAR_BRANCH_NUMERIC_CANDIDATE" if solved else
            "LARGE_BOX_NONCOPLANAR_DIRECTION_NONLINEARLY_OBSTRUCTED"
            if obstructed else
            "LARGE_BOX_NONCOPLANAR_SOLVABILITY_INCONCLUSIVE"),
        "claim_boundary": (
            "Range correction at the canonical rest Hessian. A numeric pass "
            "still requires augmented solve, stability, transport, and box "
            "refinement; obstruction is local to this unique direction."),
    }
    output = HERE / "canonical_noncoplanar_large_box_solvability.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE / "canonical_noncoplanar_large_box_solvability.npz",
        rest=rest, direction=direction, fractions=np.asarray(FRACTIONS),
        states=np.asarray(states))
    print(json.dumps({
        "verdict": result["verdict"],
        "rows": [{key: row[key] for key in (
            "fraction_of_rest_norm", "final_total_rms",
            "final_range_norm", "final_null_solvability_norm",
            "physical_direction_projection", "halo_psi_power_fraction")}
            for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
