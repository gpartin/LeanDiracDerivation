"""Enumerate the independent positive square-face identities of W.

The saved unstable modes are used only as inverse-design oracles.  No mode,
corner label, selected axis, or Dirac object enters any candidate action.
Every candidate is summed over the three unoriented cubic face planes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp
from scipy.optimize import linprog

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_large_box_audit as large
import canonical_noncoplanar_box_transfer as box


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., :, None]*right[..., None, :]


def shift(value: torch.Tensor, axis: int) -> torch.Tensor:
    offset = tuple(1 if entry == axis else 0 for entry in range(3))
    return torch.roll(value, shifts=offset, dims=(0, 1, 2))


def defects(raw: torch.Tensor, kind: str) -> list[torch.Tensor]:
    psi, chi = box.unpack(raw, SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)/small.CHI0
    result = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        wa, wb = shift(w, first), shift(w, second)
        wab = shift(wa, second)
        if kind == "diagonal":
            value = outer(w, wab)-outer(wa, wb)
        elif kind == "diagonal_reversed":
            value = outer(w, wab)-outer(wb, wa)
        elif kind == "diagonal_conjugate":
            value = outer(w, torch.conj(wab))-outer(wa, torch.conj(wb))
        elif kind == "diagonal_conjugate_reversed":
            value = outer(w, torch.conj(wab))-outer(wb, torch.conj(wa))
        elif kind == "row_first":
            value = outer(w, wa)-outer(wb, wab)
        elif kind == "row_second":
            value = outer(w, wb)-outer(wa, wab)
        elif kind == "edge_commutator":
            da, db = wa-w, wb-w
            value = outer(da, db)-outer(db, da)
        else:
            raise ValueError(kind)
        result.append(value)
    return result


def energy(raw: torch.Tensor, kind: str, part: str = "full") -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for value in defects(raw, kind):
        if part == "symmetric":
            value = (value+value.transpose(-1, -2))/2
        elif part == "antisymmetric":
            value = (value-value.transpose(-1, -2))/2
        result += torch.sum(torch.real(torch.conj(value)*value))/2
    return result


CANDIDATES = [
    ("diagonal_full", "diagonal", "full"),
    ("diagonal_symmetric", "diagonal", "symmetric"),
    ("diagonal_antisymmetric", "diagonal", "antisymmetric"),
    ("diagonal_reversed_full", "diagonal_reversed", "full"),
    ("diagonal_conjugate_full", "diagonal_conjugate", "full"),
    ("diagonal_conjugate_reversed_full", "diagonal_conjugate_reversed", "full"),
    ("row_first_full", "row_first", "full"),
    ("row_second_full", "row_second", "full"),
    ("edge_commutator_full", "edge_commutator", "full"),
]


def hessian_images(function, state: torch.Tensor,
                   mode: np.ndarray) -> tuple[np.ndarray, float, float]:
    gradient = grad(function)
    images, curvature = [], 0.0
    for direction in (np.real(mode), np.imag(mode)):
        vector = torch.tensor(direction, dtype=torch.float64)
        _, image = jvp(gradient, (state,), (vector,))
        image_np = image.detach().numpy()
        images.append(image_np)
        curvature += float(np.dot(direction, image_np))
    image = np.concatenate(images)
    norm = float(np.sqrt(np.vdot(mode, mode).real))
    alignment = abs(curvature)/max(norm*np.linalg.norm(image), 1e-300)
    return image, curvature/(norm**2), alignment


def main() -> None:
    state = torch.tensor(np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"],
        dtype=torch.float64)
    paths = {
        "mixed_mode": HERE/
            "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_leading_mode.npz",
        "exterior_mode": HERE/
            "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_loop_curvature_0p3_leading_mode.npz",
    }
    modes = {name: np.load(path)["phase_space_vector"][:state.numel()]
             for name, path in paths.items()}
    rows, accepted_images = [], []
    for name, kind, part in CANDIDATES:
        function = lambda raw, k=kind, p=part: energy(raw, k, p)
        value = float(function(state).detach())
        force = grad(function)(state)
        force_norm = float(torch.linalg.vector_norm(force).detach())
        responses, combined = {}, []
        for mode_name, mode in modes.items():
            image, curvature, alignment = hessian_images(function, state, mode)
            combined.append(image)
            responses[mode_name] = {
                "curvature": curvature,
                "oracle_alignment": alignment,
                "image_norm_per_mode_norm": float(
                    np.linalg.norm(image)/np.sqrt(np.vdot(mode, mode).real)),
            }
        exact = value < 1e-18 and force_norm < 1e-9
        rows.append({
            "candidate": name,
            "positive_square": True,
            "rest_energy": value,
            "rest_force_norm": force_norm,
            "zero_on_rest": exact,
            "responses": responses,
        })
        if exact:
            accepted_images.append((name, np.concatenate(combined)))

    normalized = np.column_stack([
        image/max(np.linalg.norm(image), 1e-300)
        for _, image in accepted_images])
    singular = np.linalg.svd(normalized, compute_uv=False)
    correlation = normalized.T@normalized
    exact_rows = [row for row in rows if row["zero_on_rest"]]
    severity = np.asarray((0.07498283977332108**2,
                           0.06635860865962026**2))
    curvature = np.asarray([
        [row["responses"][mode]["curvature"] for row in exact_rows]
        for mode in ("mixed_mode", "exterior_mode")])
    objective = np.r_[np.zeros(len(exact_rows)), -1.0]
    inequalities = np.column_stack((-curvature/severity[:, None],
                                    np.ones(2)))
    cone = linprog(
        objective, A_ub=inequalities, b_ub=np.zeros(2),
        A_eq=np.asarray([[1.0]*len(exact_rows)+[0.0]]), b_eq=[1.0],
        bounds=[(0.0, None)]*len(exact_rows)+[(None, None)], method="highs")
    cone_weights = {row["candidate"]: float(weight)
                    for row, weight in zip(exact_rows, cone.x[:-1], strict=True)
                    if weight > 1e-10}
    result = {
        "status": "V4_FACE_IDENTITY_BASIS_DERIVATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "zero_on_rest_candidates": [name for name, _ in accepted_images],
        "normalized_joint_response_singular_values": singular.tolist(),
        "normalized_joint_response_rank": int(np.sum(singular > 1e-8)),
        "normalized_response_correlation": correlation.tolist(),
        "positive_cone_objective": (
            "maximize the weaker curvature divided by the measured squared "
            "growth of the two unstable families at fixed unit total weight"),
        "positive_cone_success": bool(cone.success),
        "positive_cone_weights": cone_weights,
        "positive_cone_equalized_score": float(cone.x[-1]),
        "positive_cone_mode_curvatures": (curvature@cone.x[:-1]).tolist(),
        "verdict": (
            "SECOND_INDEPENDENT_ZERO_ON_REST_FACE_RESPONSE_FOUND"
            if np.sum(singular > 1e-8) >= 2 else
            "FACE_IDENTITY_BASIS_REMAINS_RANK_ONE"),
        "claim_boundary": (
            "Hessian response basis at the retained exact rest orbit. This "
            "does not select coefficients or prove spectral stability."),
    }
    (HERE/"personal_touch_face_identity_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
