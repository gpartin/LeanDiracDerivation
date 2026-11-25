"""Inverse screen for loop-resolved native Psi--chi frame actions.

The leading unresolved Floquet mode of the best exact-rest candidate is a
localized mixed-field twist/shear.  A formal rank-one projector onto that
mode would be the most direct stabilizer, but it would encode the answer and
is therefore forbidden.  This script uses that projector only as an oracle:
it asks whether generic, positive, cubic-symmetric plaquette curvatures of
the native fields produce a Hessian response aligned with the failing mode.

Every candidate is a position-only action term, is summed over all three
unoriented coordinate planes, and uses no prepared ray, corner label, Dirac
matrix, selected copy, or transport direction.  Unit coefficients are basis
screens only; no action is admitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import canonical_noncoplanar_large_box_audit as large
import canonical_noncoplanar_box_transfer as box
import plaquette_factorization_completion_probe as factorization


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def dyad(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., :, None] * torch.conj(right[..., None, :])


def norm_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value) * value, dim=(-2, -1)))


def shift(value: torch.Tensor, axis: int) -> torch.Tensor:
    offsets = [0, 0, 0]
    offsets[axis] = 1
    return torch.roll(value, shifts=tuple(offsets), dims=(0, 1, 2))


def matrix_face_curvature(field: torch.Tensor,
                          adjoint_second: bool) -> torch.Tensor:
    """Positive curvature of two one-bond changes around every face.

    For a phase-bearing cross dyad, ``Da Db^dagger-Db Da^dagger`` is phase
    invariant.  For a Hermitian dyad the corresponding commutator is used.
    Both vanish whenever the two local frame changes commute, but respond to
    a torsional mismatch accumulated around a square face.
    """
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        da = shift(field, first) - field
        db = shift(field, second) - field
        if adjoint_second:
            da_db = da @ torch.conj(db).transpose(-1, -2)
            db_da = db @ torch.conj(da).transpose(-1, -2)
        else:
            da_db = da @ db
            db_da = db @ da
        curvature = da_db - db_da
        result = result + torch.sum(norm_sq(curvature)) / 2
    return result


def cross_frame_loop(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return matrix_face_curvature(dyad(psi, chi), True)


def ordered_frame_loop(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    frame = dyad(psi, psi) + dyad(chi, chi)
    return matrix_face_curvature(frame, False)


def separated_frame_loop(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    return (matrix_face_curvature(dyad(psi, psi), False)
            + matrix_face_curvature(dyad(chi, chi), False))


def area_frame_loop(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    return matrix_face_curvature(dyad(area, area), False)


CANDIDATES = {
    "cross_frame_plaquette_curvature": cross_frame_loop,
    "combined_ordered_frame_plaquette_curvature": ordered_frame_loop,
    "separated_ordered_frame_plaquette_curvature": separated_frame_loop,
    "area_frame_plaquette_curvature": area_frame_loop,
    "W_opposite_corner_factorization": factorization.plaquette_potential,
}


def mode_response(function, state: torch.Tensor,
                  mode: np.ndarray) -> dict[str, float]:
    gradient = grad(function)
    norm_sq_mode = float(np.vdot(mode, mode).real)
    curvature = 0.0
    image_norm_sq = 0.0
    overlap = 0.0
    for quadrature in (np.real(mode), np.imag(mode)):
        vector = torch.tensor(quadrature, dtype=torch.float64)
        if float(torch.dot(vector, vector)) < 1e-30:
            continue
        _, image = jvp(gradient, (state,), (vector,))
        image_np = image.detach().numpy()
        curvature += float(torch.dot(vector, image).detach())
        image_norm_sq += float(np.dot(image_np, image_np))
        overlap += float(np.dot(quadrature, image_np))
    image_norm = float(np.sqrt(image_norm_sq))
    oracle_alignment = abs(overlap) / max(
        np.sqrt(norm_sq_mode) * image_norm, 1e-300)
    return {
        "quadrature_averaged_Hessian_curvature": curvature / norm_sq_mode,
        "Hessian_image_norm_per_mode_norm": image_norm / np.sqrt(norm_sq_mode),
        "rank_one_oracle_alignment": oracle_alignment,
    }


def main() -> None:
    state = torch.tensor(np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"],
        dtype=torch.float64)
    saved_paths = {
        "pre_loop_mixed_Psi_chi_mode": HERE /
            "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_leading_mode.npz",
        "post_loop_Psi_exterior_mode": HERE /
            "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_loop_curvature_0p3_leading_mode.npz",
    }
    saved_modes = {name: np.load(path) for name, path in saved_paths.items()}
    rows = []
    for name, function in CANDIDATES.items():
        value = function(state)
        gradient = grad(function)(state)
        row = {
            "candidate": name,
            "positive_square": True,
            "cubic_plane_sum": True,
            "energy_at_exact_rest": float(value.detach()),
            "rest_force_norm": float(torch.linalg.vector_norm(gradient).detach()),
            "rest_force_rms": float(
                torch.linalg.vector_norm(gradient).detach()
                / np.sqrt(state.numel())),
        }
        row["mode_responses"] = {
            label: mode_response(
                function, state, saved["phase_space_vector"][:state.numel()])
            for label, saved in saved_modes.items()
        }
        rows.append(row)
    rows.sort(key=lambda row: row["mode_responses"]
              ["pre_loop_mixed_Psi_chi_mode"]["rank_one_oracle_alignment"],
              reverse=True)
    result = {
        "status": "V4_LOOP_CURVATURE_INVERSE_ACTION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "failing_multipliers": {
            label: {
                "real": complex(saved["eigenvalue"]).real,
                "imag": complex(saved["eigenvalue"]).imag,
                "modulus": abs(complex(saved["eigenvalue"])),
            }
            for label, saved in saved_modes.items()
        },
        "inverse_requirement": (
            "positive restoring curvature on the cubic orbit of the mixed "
            "Psi-chi carrier twist/shear, with rigid phase and translation "
            "directions retained as null or symmetry directions"),
        "forbidden_oracle": (
            "the rank-one Hessian projector onto the saved unstable mode; "
            "used only to score alignment and never installed in an action"),
        "candidate_class": (
            "lowest local positive plaquette commutator curvatures of native "
            "Psi/chi dyads, summed over xy, xz, and yz planes"),
        "rows": rows,
        "best_screen_candidate": rows[0]["candidate"],
        "verdict": (
            "LOOP_ACTION_CLASS_HAS_NONZERO_REQUIRED_RESPONSE"
            if rows[0]["mode_responses"]["pre_loop_mixed_Psi_chi_mode"]
                ["rank_one_oracle_alignment"] > 1e-6
            and abs(rows[0]["mode_responses"]["pre_loop_mixed_Psi_chi_mode"]
                    ["quadrature_averaged_Hessian_curvature"]) > 1e-12
            else "LOOP_ACTION_CLASS_MISSES_REQUIRED_RESPONSE"),
        "claim_boundary": (
            "A nonzero aligned Hessian response only earns a Fredholm rest-"
            "branch and full Floquet test. It is not coefficient provenance, "
            "an admitted action, stability, movement, or Dirac closure."),
    }
    output = HERE / "personal_touch_loop_curvature_screen.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
