"""Test native-force closure of the overlapping-cube localized embedding.

This is the cheap action gate after ``localized_overlapping_cube_embedding``.
Each flavor copy is mapped into the actual Psi/chi carrier sites using the
carrier's native ray and companion frames.  The complete real-linear native
acceleration tangent then acts on that eight-real-dimensional subspace.  The
reported cross-block is (I-QQ^T) A Q, including every component and exterior
site on the side-three periodic box.

The relative complex chi response is selected only by minimizing native
cross-block leakage; no Dirac comparator is used.  A large irreducible leak
means the spatial rank result needs a halo/velocity graph or a different
native action before it can be a closed localized spinor sector.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import cubical_edge_shift_channels as cubical
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 3


def core_sites() -> list[tuple[int, int, int]]:
    center = SIDE // 2 - 1
    return [tuple(center + bit for bit in corner)
            for corner in itertools.product((0, 1), repeat=3)]


def flavor_bases() -> dict[str, np.ndarray]:
    identity = np.eye(8, dtype=complex)
    flavor = cubical.flavor()
    result = {}
    for name, projector in (
            ("plus", (identity + flavor) / 2),
            ("minus", (identity - flavor) / 2)):
        values, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
        result[name] = vectors[:, values > 1e-11]
    return result


def carrier_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    sites = core_sites()
    psi_values = np.stack([psi[site] for site in sites])
    chi_values = np.stack([chi[site] for site in sites])
    return (psi, chi,
            psi_values / np.linalg.norm(psi_values, axis=1)[:, None],
            chi_values / np.linalg.norm(chi_values, axis=1)[:, None])


def real_embedding(basis: np.ndarray, psi_frame: np.ndarray,
                   chi_frame: np.ndarray, chi_scale: complex) -> np.ndarray:
    columns = []
    sites = core_sites()
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, site in enumerate(sites):
            amplitude = basis[corner, component]
            dpsi[site] = amplitude * psi_frame[corner]
            dchi[site] = chi_scale * amplitude * chi_frame[corner]
        columns.append(rank.pack(dpsi, dchi))
        columns.append(rank.pack(1j * dpsi, 1j * dchi))
    raw = np.column_stack(columns)
    q, r = np.linalg.qr(raw)
    diagonal = abs(np.diag(r))
    return q[:, diagonal > 1e-11 * max(float(diagonal[0]), 1.0)]


def closure(matrix: np.ndarray, q: np.ndarray) -> dict:
    response = matrix @ q
    retained = q @ (q.T @ response)
    leakage = response - retained
    response_norm = float(np.linalg.norm(response))
    leakage_norm = float(np.linalg.norm(leakage))
    return {
        "embedding_real_rank": int(q.shape[1]),
        "native_response_norm": response_norm,
        "cross_block_norm": leakage_norm,
        "relative_cross_block": leakage_norm / max(response_norm, 1e-30),
    }


def leakage_fraction(matrix: np.ndarray, basis: np.ndarray,
                     psi_frame: np.ndarray, chi_frame: np.ndarray,
                     parameters: np.ndarray) -> float:
    chi_scale = parameters[0] + 1j * parameters[1]
    q = real_embedding(basis, psi_frame, chi_frame, chi_scale)
    return closure(matrix, q)["relative_cross_block"]


def site_channel_decomposition(vector: np.ndarray,
                               psi_frame: np.ndarray,
                               chi_frame: np.ndarray) -> dict:
    core = np.zeros((SIDE, SIDE, SIDE), dtype=bool)
    for site in core_sites():
        core[site] = True
    psi_power = np.zeros((SIDE, SIDE, SIDE))
    chi_power = np.zeros_like(psi_power)
    frame_power = {
        "psi_ray": 0.0, "psi_companion": 0.0, "psi_normal": 0.0,
        "chi_ray": 0.0, "chi_companion": 0.0, "chi_normal": 0.0,
    }
    columns = vector.T if vector.ndim == 2 else vector[None, :]
    for column in columns:
        psi, chi = rank.native.unpack(column, SIDE)
        psi_power += rank.native.norm2(psi)
        chi_power += rank.B * rank.native.norm2(chi)
        for corner, site in enumerate(core_sites()):
            u = psi_frame[corner]
            v = chi_frame[corner]
            frame_power["psi_ray"] += abs(np.vdot(u, psi[site]))**2
            frame_power["psi_companion"] += abs(np.vdot(v, psi[site]))**2
            frame_power["psi_normal"] += abs(psi[site][2])**2
            frame_power["chi_ray"] += rank.B * abs(np.vdot(u, chi[site]))**2
            frame_power["chi_companion"] += rank.B * abs(np.vdot(v, chi[site]))**2
            frame_power["chi_normal"] += rank.B * abs(chi[site][2])**2
    total = max(float(np.sum(psi_power) + np.sum(chi_power)), 1e-30)
    frame_total = max(sum(frame_power.values()), 1e-30)
    return {
        "psi_fraction": float(np.sum(psi_power) / total),
        "chi_fraction": float(np.sum(chi_power) / total),
        "core_fraction": float(
            (np.sum(psi_power[core]) + np.sum(chi_power[core])) / total),
        "halo_exterior_fraction": float(
            (np.sum(psi_power[~core]) + np.sum(chi_power[~core])) / total),
        "core_native_frame_fractions": {
            name: float(value / frame_total)
            for name, value in frame_power.items()
        },
    }


def classify(name: str, basis: np.ndarray, matrix: np.ndarray,
             psi_frame: np.ndarray, chi_frame: np.ndarray) -> dict:
    bounds = [(-4, 4), (-4, 4)]
    global_fit = differential_evolution(
        lambda value: leakage_fraction(
            matrix, basis, psi_frame, chi_frame, value),
        bounds, seed=40411 + (0 if name == "plus" else 1),
        maxiter=120, popsize=12, tol=1e-10, polish=False)
    local_fit = minimize(
        lambda value: leakage_fraction(
            matrix, basis, psi_frame, chi_frame, value),
        global_fit.x, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-12, "fatol": 1e-12})
    chi_scale = complex(local_fit.x[0], local_fit.x[1])
    q = real_embedding(basis, psi_frame, chi_frame, chi_scale)
    response = matrix @ q
    leakage = response - q @ (q.T @ response)
    row = closure(matrix, q)
    row.update({
        "flavor": name,
        "native_selected_chi_scale_real": float(chi_scale.real),
        "native_selected_chi_scale_imag": float(chi_scale.imag),
        "optimizer_success": bool(local_fit.success),
        "leakage_decomposition": site_channel_decomposition(
            leakage, psi_frame, chi_frame),
    })
    controls = {}
    for control_name, control_scale in (
            ("psi_only", 0j), ("equal_joint", 1 + 0j),
            ("opposite_joint", -1 + 0j), ("quadrature_joint", 1j)):
        control_q = real_embedding(
            basis, psi_frame, chi_frame, control_scale)
        controls[control_name] = closure(matrix, control_q)[
            "relative_cross_block"]
    row["controls"] = controls
    return row


def main() -> None:
    rank.lock_inertia()
    psi, chi, psi_frame, chi_frame = carrier_frames()
    tangent = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)[0]
    rows = [classify(name, basis, tangent, psi_frame, chi_frame)
            for name, basis in flavor_bases().items()]
    best = min(row["relative_cross_block"] for row in rows)
    result = {
        "status": "V4_LOCALIZED_EMBEDDING_NATIVE_FORCE_CLOSURE_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "native_tangent": "complete canonical position-force tangent",
        "rows": rows,
        "best_relative_cross_block": float(best),
        "verdict": (
            "CORE_ONLY_NATIVE_FORCE_CLOSURE_PASS"
            if best < 1e-8 else
            "CORE_ONLY_EMBEDDING_REQUIRES_HALO_OR_ACTION_REPAIR"),
        "root_cause_if_failed": (
            "The joint Psi/chi spatial map has the required rank and exact "
            "translation equivariance, but a core-only carrier-derived graph "
            "is not an invariant subspace of the full native force. The "
            "leakage decomposition distinguishes missing halo response from "
            "an internal Psi/chi mismatch."),
        "claim_boundary": (
            "This is a complete real configuration-force cross-block on the "
            "side-three box. It does not include the implicit velocity metric "
            "or prove a two-layer positive-frequency sector."),
        "corrective_action": (
            "If leakage is halo dominated, solve the native invariant-graph "
            "equation for the first shell before changing the action. If it "
            "is core dominated, use the residual as the next action-response "
            "target. Then repeat with both committed layers."),
    }
    (HERE / "localized_embedding_native_closure.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
