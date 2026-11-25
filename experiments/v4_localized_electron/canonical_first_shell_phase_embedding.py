"""Derive a full-shell, two-layer localized tangent from canonical V40.

The spatial seed is the already-audited joint Psi/chi cubical embedding.  No
Dirac matrix or target motion is used.  The canonical centered update is
linearized about the exact localized rest orbit in its rotating frame.  The
four positive-phase native modes with greatest overlap with each flavor seed
are selected, and their realification gives an eight-real-dimensional
candidate phase-space sector including every exterior site and both committed
time layers.

The gate requires more than formal invariance of eigenvectors: the current
layer must have rank eight, retain substantial overlap with the seed, and
remain localized on the carrier plus its C19 first shell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import edge_snap_action_rank as rank
import localized_embedding_native_closure as local


HERE = Path(__file__).resolve().parent
SIDE = 3
DT = 1.0 / 256.0


def canonical_rotating_monodromy() -> tuple[np.ndarray, dict]:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    acceleration = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)[0]
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    phase = np.exp(1j * theta)
    rotation = rank.native.rotation_matrix(SIDE, phase, 1.0 + 0.0j, 1)
    inverse = rotation.T
    dimension = acceleration.shape[0]
    forward = inverse @ (2 * np.eye(dimension) + DT**2 * acceleration)
    backward = -(inverse @ inverse)
    monodromy = np.block([
        [forward, backward],
        [np.eye(dimension), np.zeros((dimension, dimension))],
    ])
    return monodromy, {
        "side": SIDE,
        "dt": DT,
        "theta": theta,
        "rest_frequency": theta / DT,
        "configuration_dimension": dimension,
        "phase_space_dimension": 2 * dimension,
    }


def realify(vectors: np.ndarray) -> np.ndarray:
    columns = []
    for column in range(vectors.shape[1]):
        columns.extend((vectors[:, column].real, vectors[:, column].imag))
    raw = np.column_stack(columns)
    q, r = np.linalg.qr(raw)
    diagonal = abs(np.diag(r))
    return q[:, diagonal > 1e-10 * max(float(diagonal[0]), 1.0)]


def phase_seed(spatial: np.ndarray) -> np.ndarray:
    # This is only a selector for the native spectral subspace.  The previous
    # layer is deliberately not assigned a Dirac phase; both layers use the
    # same spatial seed and the eigenproblem derives their relation.
    raw = np.vstack((spatial, spatial))
    q, _ = np.linalg.qr(raw)
    return q[:, :spatial.shape[1]]


def site_power(current_columns: np.ndarray) -> np.ndarray:
    result = np.zeros((SIDE, SIDE, SIDE))
    for column in range(current_columns.shape[1]):
        psi, chi = rank.native.unpack(current_columns[:, column], SIDE)
        result += np.sum(abs(psi)**2, axis=-1)
        result += rank.B * np.sum(abs(chi)**2, axis=-1)
    return result


def classify(flavor: str, basis: np.ndarray, monodromy: np.ndarray,
             metadata: dict, psi_frame: np.ndarray,
             chi_frame: np.ndarray, chi_scale: complex) -> dict:
    spatial = local.real_embedding(
        basis, psi_frame, chi_frame, chi_scale)
    selector = phase_seed(spatial)
    values, left_vectors = eig(monodromy, check_finite=False)
    current_dimension = metadata["configuration_dimension"]
    phases = np.angle(values)
    positive = np.flatnonzero(phases > 1e-9)
    scores = []
    for index in positive:
        vector = left_vectors[:, index]
        norm = max(float(np.vdot(vector, vector).real), 1e-300)
        score = float(np.linalg.norm(selector.conj().T @ vector)**2 / norm)
        scores.append((score, index))
    theta = metadata["theta"]
    rest_band = [(score, index) for score, index in scores
                 if abs(phases[index] - theta) <= theta / 2]
    selected = [index for _score, index in sorted(
        rest_band, reverse=True)[:4]]
    if len(selected) < 4:
        # Preserve a deterministic diagnostic instead of fabricating the
        # missing rest-frequency coordinates from unrelated high-frequency
        # internal modes.
        selected = [index for _score, index in sorted(rest_band, reverse=True)]
    if not selected:
        nearest = sorted(
            positive, key=lambda index: abs(phases[index] - theta))[:8]
        return {
            "flavor": flavor,
            "chi_scale_selector": {
                "real": float(chi_scale.real),
                "imag": float(chi_scale.imag)},
            "selected_native_modes": [],
            "rest_band_candidate_count": 0,
            "nearest_positive_phase_modes": [{
                "phase": float(phases[index]),
                "frequency": float(phases[index] / DT),
                "modulus": float(abs(values[index])),
                "distance_from_rest_phase": float(abs(phases[index] - theta)),
            } for index in nearest],
            "phase_sector_real_rank": 0,
            "current_layer_real_rank": 0,
            "invariance_relative_residual": None,
            "seed_capture_fraction": 0.0,
            "seed_current_principal_values": [],
            "core_power_fraction": None,
            "first_shell_power_fraction": None,
            "passes_localized_phase_embedding_gate": False,
        }
    complex_sector = left_vectors[:, selected]
    sector = realify(complex_sector)
    retained = sector @ (sector.T @ (monodromy @ sector))
    invariance = float(np.linalg.norm(monodromy @ sector - retained)
                       / max(np.linalg.norm(monodromy @ sector), 1e-300))
    current = sector[:current_dimension]
    current_rank = int(np.linalg.matrix_rank(current, tol=1e-10))
    current_q, _ = np.linalg.qr(current)
    spatial_overlap = np.linalg.svd(
        spatial.T @ current_q[:, :current_rank], compute_uv=False)
    seed_capture = float(np.linalg.norm(current_q[:, :current_rank].T @ spatial)**2
                         / max(spatial.shape[1], 1))

    power = site_power(current)
    total = max(float(np.sum(power)), 1e-300)
    core_mask = np.zeros_like(power, dtype=bool)
    for site in local.core_sites():
        core_mask[site] = True
    core_fraction = float(np.sum(power[core_mask]) / total)
    shell_fraction = float(np.sum(power[~core_mask]) / total)
    eigen_rows = [{
        "real": float(values[index].real),
        "imag": float(values[index].imag),
        "modulus": float(abs(values[index])),
        "phase": float(np.angle(values[index])),
        "frequency": float(np.angle(values[index]) / DT),
        "selector_overlap_score": float(dict(
            (candidate, score) for score, candidate in scores)[index]),
    } for index in selected]
    passes = (
        len(selected) == 4 and sector.shape[1] == 8 and current_rank == 8
        and invariance < 1e-8 and seed_capture > 0.5
        and core_fraction > 0.5 and shell_fraction > 1e-8)
    return {
        "flavor": flavor,
        "chi_scale_selector": {
            "real": float(chi_scale.real), "imag": float(chi_scale.imag)},
        "selected_native_modes": eigen_rows,
        "rest_band_candidate_count": len(rest_band),
        "phase_sector_real_rank": int(sector.shape[1]),
        "current_layer_real_rank": current_rank,
        "invariance_relative_residual": invariance,
        "seed_capture_fraction": seed_capture,
        "seed_current_principal_values": [
            float(value) for value in spatial_overlap],
        "core_power_fraction": core_fraction,
        "first_shell_power_fraction": shell_fraction,
        "passes_localized_phase_embedding_gate": passes,
    }


def main() -> None:
    rank.lock_inertia()
    monodromy, metadata = canonical_rotating_monodromy()
    _psi, _chi, psi_frame, chi_frame = local.carrier_frames()
    bases = local.flavor_bases()
    # These scales are the native minimum-leakage selectors from the canonical
    # configuration-force audit, not Dirac fits.
    scales = {
        "plus": 0.008305467684459727 + 5.0613584453970665e-09j,
        "minus": 0.00838891677437946 - 1.200406309785864e-08j,
    }
    rows = [classify(name, basis, monodromy, metadata, psi_frame, chi_frame,
                     scales[name]) for name, basis in bases.items()]
    result = {
        "status": "V4_CANONICAL_FIRST_SHELL_PHASE_EMBEDDING",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "native_action": "canonical V40 position action and centered update",
        "metadata": metadata,
        "selection_rule": (
            "four native modes inside one-half rest-phase of the exact "
            "carrier clock, ranked by overlap with the pre-existing joint "
            "localized spatial seed; no Dirac comparator"),
        "rows": rows,
        "verdict": (
            "CANONICAL_FULL_SHELL_PHASE_EMBEDDING_PASS"
            if all(row["passes_localized_phase_embedding_gate"] for row in rows)
            else "CANONICAL_FULL_SHELL_PHASE_EMBEDDING_FAIL"),
        "claim_boundary": (
            "Linear tangent invariant-sector and localization gate only. It "
            "does not prove a nonlinear translating branch or Dirac current."),
    }
    output = HERE / "canonical_first_shell_phase_embedding.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE / "canonical_first_shell_phase_embedding.npz",
        monodromy=monodromy)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
