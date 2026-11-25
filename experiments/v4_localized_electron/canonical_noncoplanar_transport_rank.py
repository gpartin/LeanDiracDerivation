"""Fixed-charge transport rank on the canonical side-five chi branch.

The branch leaves the eight-site Psi pattern unchanged while rotating complex
chi.  This audit uses the exact canonical interval action and its Noether
densities to test whether that native chi orientation supplies the missing
third carrier-center response.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import localized_embedding_native_closure as embedding


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE
right_momentum = grad(lambda left, right: box.interval(left, right, SIDE),
                      argnums=1)


def density_tensor(left: torch.Tensor, right: torch.Tensor,
                   mode: str) -> torch.Tensor:
    momentum = right_momentum(left, right)
    psi, chi = box.unpack(right, SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator = torch.stack((
        torch.real(1j*psi if mode == "Psi" else zero_psi),
        torch.imag(1j*psi if mode == "Psi" else zero_psi),
        torch.real(1j*chi if mode == "chi" else zero_chi),
        torch.imag(1j*chi if mode == "chi" else zero_chi)), dim=-2).reshape(-1)
    return (momentum*generator).reshape(
        SIDE, SIDE, SIDE, 4, 3).sum(dim=(-1, -2))


def density_derivative(left: np.ndarray, right: np.ndarray,
                       dleft: np.ndarray, dright: np.ndarray,
                       mode: str) -> tuple[np.ndarray, np.ndarray]:
    l = torch.tensor(left, dtype=torch.float64)
    r = torch.tensor(right, dtype=torch.float64)
    dl = torch.tensor(dleft, dtype=torch.float64)
    dr = torch.tensor(dright, dtype=torch.float64)
    value, derivative = jvp(
        lambda a, b: density_tensor(a, b, mode), (l, r), (dl, dr))
    return value.detach().numpy(), derivative.detach().numpy()


def center_derivative(density: np.ndarray,
                      derivative: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    dtotal = float(np.sum(derivative))
    coordinates = np.indices(density.shape)
    result = []
    for axis in range(3):
        moment = float(np.sum(coordinates[axis]*density))
        dmoment = float(np.sum(coordinates[axis]*derivative))
        result.append((dmoment*total-moment*dtotal)/total**2)
    return np.asarray(result)


def core_frames(state: np.ndarray):
    psi, chi = large.unpack_np(state)
    origin = SIDE//2-1
    sites = [tuple(origin+bit for bit in corner)
             for corner in itertools.product((0, 1), repeat=3)]
    p = np.asarray([psi[site] for site in sites])
    c = np.asarray([chi[site] for site in sites])
    return (p/np.linalg.norm(p, axis=1)[:, None],
            c/np.linalg.norm(c, axis=1)[:, None])


def placed(amplitudes: np.ndarray, offset: tuple[int, int, int],
           psi_frame: np.ndarray, chi_frame: np.ndarray,
           psi_phase: complex) -> np.ndarray:
    psi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
    chi = np.zeros_like(psi)
    origin = SIDE//2-1
    for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
        site = tuple(origin+offset[axis]+bits[axis] for axis in range(3))
        psi[site] += amplitudes[corner]*psi_phase*psi_frame[corner]
        chi[site] += amplitudes[corner]*chi_frame[corner]
    return box.pack(1j*psi, 1j*chi)


def targets(state: np.ndarray, axis: int, phase: complex):
    psi_frame, chi_frame = core_frames(state)
    flavors = embedding.flavor_bases()
    plus = tuple(1 if index == axis else 0 for index in range(3))
    minus = tuple(-1 if index == axis else 0 for index in range(3))
    labels, columns = [], []
    for flavor in ("plus", "minus"):
        for component in range(4):
            amplitudes = flavors[flavor][:, component]
            columns.append((placed(amplitudes, plus, psi_frame, chi_frame,
                                   phase)
                            - placed(amplitudes, minus, psi_frame, chi_frame,
                                     phase))/2)
            labels.append(f"{flavor}_{component}")
    return labels, np.column_stack(columns)


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(launch)
    current_psi, dcurrent_psi = density_derivative(
        previous, current, launch, zero, "Psi")
    following_psi, dfollowing_psi = density_derivative(
        current, following, zero, target, "Psi")
    _current_chi, dcurrent_chi = density_derivative(
        previous, current, launch, zero, "chi")
    _following_chi, dfollowing_chi = density_derivative(
        current, following, zero, target, "chi")
    charges = np.asarray((
        (np.sum(dcurrent_psi)+np.sum(dfollowing_psi))/2,
        (np.sum(dcurrent_chi)+np.sum(dfollowing_chi))/2,
    ))
    displacement = (center_derivative(following_psi, dfollowing_psi)
                    - center_derivative(current_psi, dcurrent_psi))
    return np.r_[charges, displacement]


def classify(state: np.ndarray, fraction: float, phase: complex) -> dict:
    current = state
    current_tensor = torch.tensor(current, dtype=torch.float64)
    previous = box.phase_rotate(
        current_tensor, np.conjugate(phase), SIDE).detach().numpy()
    following = box.phase_rotate(
        current_tensor, phase, SIDE).detach().numpy()
    axis_rows = []
    for axis in range(3):
        labels, target_matrix = targets(state, axis, phase)
        # For the canonical position action, both off-diagonal DEL blocks are
        # exactly -M/dt, hence the compatible previous-layer launch is the
        # negative of the following-layer target.
        launch_matrix = -target_matrix
        responses = np.column_stack([
            response(previous, current, following,
                     launch_matrix[:, index], target_matrix[:, index])
            for index in range(target_matrix.shape[1])])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        fixed_charge = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        center = responses[2:]@fixed_charge
        singular = np.linalg.svd(center, compute_uv=False)
        threshold = 1e-8*max(singular[0], 1e-30)
        axis_rows.append({
            "axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [float(x)
                                                     for x in singular],
            "relative_third_singular_value": float(
                singular[2]/max(singular[0], 1e-300)),
            "response_matrix": responses.tolist(),
        })
    return {
        "fraction_of_rest_norm": fraction,
        "axis_rows": axis_rows,
        "all_axis_ranks": [row["fixed_charge_center_rank"]
                           for row in axis_rows],
        "minimum_relative_third_singular_value": float(min(
            row["relative_third_singular_value"] for row in axis_rows)),
    }


def main() -> None:
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    rest = archive["rest"]
    fractions = archive["fractions"]
    states = archive["states"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    selected = [(0.0, rest)] + [
        (float(fraction), state)
        for fraction, state in zip(fractions, states, strict=True)
        if fraction in (1e-4, 1e-3, 3e-3)]
    rows = [classify(state, fraction, phase)
            for fraction, state in selected]
    passing = [row for row in rows[1:]
               if row["all_axis_ranks"] == [3, 3, 3]
               and row["minimum_relative_third_singular_value"] > 1e-6]
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_TRANSPORT_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "passing_fractions": [row["fraction_of_rest_norm"]
                              for row in passing],
        "verdict": (
            "CANONICAL_CHI_BRANCH_SUPPLIES_THREE_AXIS_TRANSPORT_RANK"
            if passing else
            "CANONICAL_CHI_BRANCH_REMAINS_TRANSPORT_RANK_DEFICIENT"),
        "claim_boundary": (
            "Linear fixed-charge Noether-center response on exact canonical "
            "rest states. A pass is not native selection, finite movement, "
            "stability, or Dirac-current equality."),
    }
    output = HERE / "canonical_noncoplanar_transport_rank.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rows": [{
            "fraction": row["fraction_of_rest_norm"],
            "ranks": row["all_axis_ranks"],
            "minimum_relative_third_singular_value": row[
                "minimum_relative_third_singular_value"]}
            for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
