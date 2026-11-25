"""Project the quiet-receiver response onto the translated carrier face.

The older quiet-receiver gate only measured the norm in a chosen exterior
half-space.  That cannot tell whether the response reconstructs the four
complex corner values, and it assumes that changing a launch phase reverses
one fixed chiral carrier.  This gate keeps those questions separate:

* native and complex-conjugate carrier branches are tested;
* both exterior faces are measured for every launch;
* the response is projected onto the exact one-cell translated Psi face;
* the orthogonal defect is retained rather than hidden by the projection.

The translated face is a measurement target only.  It is never inserted into
the action or the native state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_noether_launch as launch
import cube_incidence_nonlinear_action as native_action
import live_clifford_bond_action as candidate


HERE = Path(__file__).resolve().parent
SIDE = 5


def pack(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack((psi.real, psi.imag, chi.real, chi.imag), dim=-2).reshape(-1)


def conjugate_state(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    return pack(torch.conj(psi), torch.conj(chi))


def fields(raw: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    psi, chi = native_box.unpack(raw, SIDE)
    return psi.detach().numpy(), chi.detach().numpy()


def translated_new_face(rest_psi: np.ndarray, axis: int, sign: int):
    """Return mask and exact Psi values newly occupied after a one-cell hop."""
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    mask = coordinate == (3 if sign > 0 else 0)
    translated = np.roll(rest_psi, sign, axis=axis)
    target = np.zeros_like(translated)
    target[mask] = translated[mask]
    return mask, target


def complex_projection(response: np.ndarray, target: np.ndarray, mask: np.ndarray):
    r = response[mask].reshape(-1)
    t = target[mask].reshape(-1)
    denominator = float(np.real(np.vdot(t, t)))
    coefficient = np.vdot(t, r)/max(denominator, 1e-300)
    fitted = coefficient*t
    response_norm = float(np.linalg.norm(r))
    target_norm = float(np.linalg.norm(t))
    defect = float(np.linalg.norm(r-fitted)/max(response_norm, 1e-300))
    overlap = float(abs(np.vdot(t, r))/(max(target_norm*response_norm, 1e-300)))
    return coefficient, response_norm, overlap, defect


def main() -> None:
    rest = native_action.expected_blocks()[0]["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    base_layers = tuple(torch.tensor(value, dtype=torch.float64) for value in (
        rotation.T@rest, rest, rotation@rest))
    rows = []
    for branch in ("native", "conjugate"):
        if branch == "native":
            previous0, current, following = base_layers
        else:
            previous0, current, following = tuple(
                conjugate_state(value) for value in base_layers)
        rest_psi, rest_chi = fields(current)
        for axis in range(3):
            for launch_sign in (-1, 1):
                for mode in ("Psi", "common"):
                    previous = launch.launch_previous(
                        previous0, axis, launch_sign, mode, 0.1)
                    full = candidate.del_residual(previous, current, following, dt)
                    baseline = candidate.incumbent.del_residual(
                        previous, current, following, dt)
                    response_psi, response_chi = fields(full-baseline)
                    for receiver_sign in (-1, 1):
                        mask, target = translated_new_face(
                            rest_psi, axis, receiver_sign)
                        coefficient, norm, overlap, defect = complex_projection(
                            response_psi, target, mask)
                        translated_chi = np.roll(
                            rest_chi, receiver_sign, axis=axis)
                        chi_target = np.zeros_like(rest_chi)
                        chi_target[mask] = (
                            translated_chi[mask]-rest_chi[mask])
                        (chi_coefficient, chi_norm, chi_overlap,
                         chi_defect) = complex_projection(
                             response_chi, chi_target, mask)
                        rows.append({
                            "carrier_branch": branch,
                            "axis": axis,
                            "launch_sign": launch_sign,
                            "launch_mode": mode,
                            "receiver_sign": receiver_sign,
                            "response_norm": norm,
                            "target_overlap": overlap,
                            "relative_orthogonal_defect": defect,
                            "projection_coefficient_real": float(coefficient.real),
                            "projection_coefficient_imag": float(coefficient.imag),
                            "chi_response_norm": chi_norm,
                            "chi_target_overlap": chi_overlap,
                            "chi_relative_orthogonal_defect": chi_defect,
                            "chi_projection_coefficient_real": float(
                                chi_coefficient.real),
                            "chi_projection_coefficient_imag": float(
                                chi_coefficient.imag),
                        })

    active = [row for row in rows if row["response_norm"] > 1e-10]
    best = sorted(active, key=lambda row: (
        -row["target_overlap"], row["relative_orthogonal_defect"]))[:12]
    result = {
        "status": "V4_LIVE_LINK_RECEIVER_PATTERN_GATE",
        "link_source": candidate.LINK_SOURCE,
        "scale": candidate.SCALE,
        "active_response_count": len(active),
        "maximum_target_overlap": max(
            (row["target_overlap"] for row in active), default=0.0),
        "minimum_active_orthogonal_defect": min(
            (row["relative_orthogonal_defect"] for row in active), default=1.0),
        "best_rows": best,
        "rows": rows,
        "verdict": (
            "LIVE_LINK_SEEDS_EXACT_TRANSLATED_FACE"
            if active and max(row["target_overlap"] for row in active) > 1-1e-10
            else "LIVE_LINK_RECEIVER_PATTERN_MISMATCH"),
        "claim_boundary": (
            "First reciprocal action residual at the quiet exterior face. "
            "The translated pattern is used only for measurement; this is "
            "not a completed update or a V4-006 pass."),
    }
    suffix = candidate.LINK_SOURCE.replace(".", "p")
    (HERE/f"live_link_receiver_pattern_gate_{suffix}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
