"""Even-periodic-box control for the live-Clifford movement action.

An odd periodic box cannot globally tile a period-two cubical cochain gauge.
This control repeats the moving-sector test on side four using the same native
eight-site carrier and the exact reciprocal action increment.  It determines
whether the apparent one-copy loss is a finite-boundary parity artifact.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import torch
from torch.func import grad, jvp

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as box
import current_action_localized_envelope_closure as envelope5
import live_clifford_bond_action as action
import live_clifford_link_geometry as geometry
import localized_stability as native
import mixed_prong_dirac_long_tangent as time_tools


HERE = Path(__file__).resolve().parent
SIDE = 4
OFFSETS = list(itertools.product((-2, -1, 0, 1), repeat=3))
CORNERS = tuple(itertools.product((0, 1), repeat=3))


def orth(columns: np.ndarray) -> np.ndarray:
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    tolerance = 1e-11*max(float(singular[0]), 1.0)
    return u[:, singular > tolerance]


def fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi, chi = native.carrier_fields(SIDE, native_time.AMPLITUDE_SQ)
    return box.pack(psi, chi), psi, chi


def rays(field: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, list]:
    sites = [tuple(int(value) for value in site)
             for site in np.argwhere(np.sum(abs(psi)**2, axis=-1) > 1e-12)]
    sites.sort()
    values = np.stack([field[site] for site in sites])
    return values/np.linalg.norm(values, axis=1)[:, None], sites


def raw_columns(basis: np.ndarray, psi_frame: np.ndarray,
                sites: list) -> np.ndarray:
    columns = []
    for offset in OFFSETS:
        for component in range(4):
            dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
            dchi = np.zeros_like(dpsi)
            for corner, site in enumerate(sites):
                target = tuple((site[axis]+offset[axis]) % SIDE
                               for axis in range(3))
                dpsi[target] += basis[corner, component]*psi_frame[corner]
            columns.extend((box.pack(dpsi, dchi),
                            box.pack(1j*dpsi, dchi)))
    return np.column_stack(columns)


def coefficients(component: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    phase = np.exp(1j*np.asarray(OFFSETS)@momentum)
    field = phase[:, None]*component
    values = []
    for row in field:
        for value in row:
            values.extend((value.real, value.imag))
    return np.asarray(values)


def main() -> None:
    rest, psi, chi = fields()
    psi_rays, sites = rays(psi, psi)
    chi_rays, _ = rays(chi, psi)
    psi_generators = [geometry.live_generator(psi_rays, axis)
                      for axis in range(3)]
    chi_generators = [geometry.live_generator(chi_rays, axis)
                      for axis in range(3)]
    intertwiner = geometry.diagonal_intertwiner(
        psi_generators, chi_generators)
    flavor = 1j*psi_generators[0]@psi_generators[1]@psi_generators[2]
    identity = np.eye(8, dtype=complex)
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = time_tools.phase_rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))

    def residual(previous, current, following):
        def summed(left, middle, right):
            return (action.one_form(left, middle, dt)
                    + action.one_form(middle, right, dt))
        return grad(summed, argnums=1)(previous, current, following)

    chi_mask = np.zeros(rest.size, dtype=bool)
    for site in range(rest.size//12):
        chi_mask[site*12+6:site*12+12] = True
    rows = []
    fundamental = 2*np.pi/SIDE
    for sector, projector in (("plus", (identity+flavor)/2),
                              ("minus", (identity-flavor)/2)):
        values, vectors = np.linalg.eigh(
            (projector+projector.conj().T)/2)
        basis = vectors[:, values > 0.5]
        raw = raw_columns(basis, psi_rays, sites)
        retained = orth(raw)
        for axis in range(3):
            alpha = basis.conj().T@psi_generators[axis]@basis
            momentum = np.zeros(3)
            momentum[axis] = fundamental
            component = np.zeros(4, dtype=complex)
            component[0] = 1
            current_c = coefficients(component, momentum)
            backward = expm(1j*dt*np.sin(fundamental)*alpha)@component
            forward = expm(-1j*dt*np.sin(fundamental)*alpha)@component
            previous_c = coefficients(backward, momentum)
            following_c = coefficients(forward, momentum)
            directions = tuple(torch.tensor(value, dtype=torch.float64)
                               for value in (rotation.T@(raw@previous_c),
                                             raw@current_c,
                                             rotation@(raw@following_c)))
            _value, image = jvp(residual, layers, directions)
            response = image.detach().numpy()
            projected = retained@(retained.T@response)
            projected[chi_mask] = response[chi_mask]
            rows.append({
                "sector": sector,
                "axis": axis,
                "retained_real_rank": int(retained.shape[1]),
                "expected_real_rank": SIDE**3*4*2,
                "response_norm": float(np.linalg.norm(response)),
                "psi_sector_cross_block_with_independent_chi": float(
                    np.linalg.norm(response-projected)
                    / max(np.linalg.norm(response), 1e-30)),
            })
    passed = all(
        row["retained_real_rank"] == row["expected_real_rank"]
        and row["psi_sector_cross_block_with_independent_chi"] < 1e-8
        for row in rows)
    result = {
        "status": "V4_LIVE_CLIFFORD_EVEN_BOX_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "periodic_parity_compatible": True,
        "link_source": action.LINK_SOURCE,
        "intertwiner_defect": float(max(np.linalg.norm(
            chi_generators[axis]@intertwiner
            - intertwiner@psi_generators[axis]) for axis in range(3))),
        "rows": rows,
        "verdict": (
            "EVEN_BOX_LIVE_CLIFFORD_BOTH_COPIES_CLOSED"
            if passed else "EVEN_BOX_LIVE_CLIFFORD_GATE_OPEN"),
        "claim_boundary": (
            "Linear periodic even-box action increment only. It does not "
            "admit the action or prove nonlinear localized motion."),
    }
    output = HERE/"live_clifford_even_box_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
