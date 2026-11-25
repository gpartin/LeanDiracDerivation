"""Test localized four-component envelopes under the current action tangent.

The spatial lift is built from the native carrier's Psi ray and chi companion
at each of the eight corners, restricted to either cubical flavor copy.  The
same four-component profile is placed at one center, the six neighboring
centers, or the complete nearest 3x3x3 center shell.  We then calculate the
cross block of the exact reduced-tight-frame acceleration tangent.

No Dirac operator, target shift, projector force, or runtime restriction is
introduced.  The projectors are used only to define and measure the derived
four-component readout.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import cubical_edge_shift_channels as cubical
import edge_snap_action_rank as rank
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


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


def carrier_frames(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    psi, chi = rank.native.unpack(rest, SIDE)
    sites = [tuple(int(value) for value in site)
             for site in np.argwhere(rank.native.norm2(psi) > 1e-12)]
    sites.sort()
    if len(sites) != 8:
        raise RuntimeError(f"expected eight carrier sites, found {len(sites)}")
    psi_values = np.stack([psi[site] for site in sites])
    chi_values = np.stack([chi[site] for site in sites])
    return (
        psi_values / np.linalg.norm(psi_values, axis=1)[:, None],
        chi_values / np.linalg.norm(chi_values, axis=1)[:, None],
        sites,
    )


def raw_columns(basis: np.ndarray, psi_frame: np.ndarray,
                chi_frame: np.ndarray, sites: list[tuple[int, ...]],
                offsets: list[tuple[int, int, int]],
                chi_scale: complex) -> np.ndarray:
    columns = []
    for offset in offsets:
        for component in range(basis.shape[1]):
            dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
            dchi = np.zeros_like(dpsi)
            for corner, site in enumerate(sites):
                target = tuple((site[axis] + offset[axis]) % SIDE
                               for axis in range(3))
                amplitude = basis[corner, component]
                dpsi[target] += amplitude * psi_frame[corner]
                dchi[target] += chi_scale * amplitude * chi_frame[corner]
            columns.extend((rank.pack(dpsi, dchi),
                            rank.pack(1j*dpsi, 1j*dchi)))
    return np.column_stack(columns)


def orth(raw: np.ndarray) -> np.ndarray:
    u, singular, _ = np.linalg.svd(raw, full_matrices=False)
    tolerance = 1e-11 * max(float(singular[0]), 1.0)
    return u[:, singular > tolerance]


def closure(matrix: np.ndarray, q: np.ndarray) -> dict:
    response = matrix @ q
    leakage = response - q @ (q.T @ response)
    return {
        "real_rank": int(q.shape[1]),
        "relative_cross_block": float(
            np.linalg.norm(leakage) / max(np.linalg.norm(response), 1e-30)),
        "response_norm": float(np.linalg.norm(response)),
        "leakage_norm": float(np.linalg.norm(leakage)),
    }


def classify(matrix: np.ndarray, basis: np.ndarray,
             psi_frame: np.ndarray, chi_frame: np.ndarray,
             sites: list[tuple[int, ...]],
             offsets: list[tuple[int, int, int]], seed: complex,
             optimize_scale: str) -> dict:
    def objective(values: np.ndarray) -> float:
        q = orth(raw_columns(
            basis, psi_frame, chi_frame, sites, offsets,
            complex(values[0], values[1])))
        return closure(matrix, q)["relative_cross_block"]

    if optimize_scale == "complex":
        fit = minimize(
            objective, np.asarray((seed.real, seed.imag)), method="Nelder-Mead",
            options={"maxiter": 160, "xatol": 1e-9, "fatol": 1e-9})
        scale = complex(fit.x[0], fit.x[1])
        optimizer_success = bool(fit.success)
    elif optimize_scale == "real":
        fit = minimize_scalar(
            lambda value: objective(np.asarray((value, 0.0))),
            bounds=(0.0, 0.01), method="bounded",
            options={"maxiter": 30, "xatol": 1e-6})
        scale = complex(fit.x, 0.0)
        optimizer_success = bool(fit.success)
    else:
        scale = seed
        optimizer_success = None
    q = orth(raw_columns(
        basis, psi_frame, chi_frame, sites, offsets, scale))
    row = closure(matrix, q)
    row.update({
        "center_count": len(offsets),
        "chi_scale_real": float(scale.real),
        "chi_scale_imag": float(scale.imag),
        "optimizer_success": optimizer_success,
    })
    return row


def main() -> None:
    rank.lock_inertia()
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE / "personal_touch_activated_bond_basis_v2_cache.npz")
    stiffness = (bare["stiffness"]
                 + additions["hessian_carrier_carrier_chi"]
                 + additions["hessian_carrier_carrier_Psi_tight_frame"])
    mass_diagonal = np.diag(bare["mass"])
    tangent = -stiffness / mass_diagonal[:, None]
    psi_frame, chi_frame, sites = carrier_frames(bare["rest"])
    center_sets = {
        "one_center": [(0, 0, 0)],
        "center_plus_faces": [(0, 0, 0),
                              (1, 0, 0), (-1, 0, 0),
                              (0, 1, 0), (0, -1, 0),
                              (0, 0, 1), (0, 0, -1)],
        "nearest_27_centers": list(itertools.product((-1, 0, 1), repeat=3)),
        "all_125_centers": list(itertools.product((-2, -1, 0, 1, 2), repeat=3)),
    }
    rows = []
    for flavor, basis in flavor_bases().items():
        seed = 1/np.sqrt(rank.B) + 0j
        cases = []
        modes = ("complex", "none", "real", "none")
        for index, (name, offsets) in enumerate(center_sets.items()):
            row = classify(
                tangent, basis, psi_frame, chi_frame, sites, offsets, seed,
                optimize_scale=modes[index])
            row["envelope"] = name
            cases.append(row)
            seed = complex(row["chi_scale_real"], row["chi_scale_imag"])
        rows.append({"flavor": flavor, "cases": cases})
    worst_27 = max(
        next(case for case in row["cases"]
             if case["envelope"] == "nearest_27_centers")["relative_cross_block"]
        for row in rows)
    worst_all = max(
        next(case for case in row["cases"]
             if case["envelope"] == "all_125_centers")["relative_cross_block"]
        for row in rows)
    result = {
        "status": "V4_CURRENT_ACTION_LOCALIZED_ENVELOPE_CLOSURE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "bare V40 plus unit carrier-carrier chi and Psi tight-frame "
            "position energies"),
        "rows": rows,
        "worst_nearest_27_center_cross_block": float(worst_27),
        "worst_all_center_cross_block": float(worst_all),
        "verdict": (
            "LOCALIZED_FOUR_COMPONENT_ENVELOPE_TANGENT_CLOSED"
            if worst_all < 1e-8 else
            "LOCALIZED_FOUR_COMPONENT_ENVELOPE_TANGENT_NOT_CLOSED"),
        "interpretation": (
            "Increasing the center envelope distinguishes a missing spatial "
            "halo from an internal field mismatch. A nonzero 27-center cross "
            "block is the part of the current action tangent that the same "
            "four-component carrier readout cannot predict."),
        "claim_boundary": (
            "This is the configuration acceleration tangent. A pass would "
            "still require both committed layers, signed native current, and "
            "nonlinear free motion. A failure is scoped to this embedding and "
            "the tested action."),
    }
    (HERE / "current_action_localized_envelope_closure.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
