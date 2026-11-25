"""Test whether canonical V40 has a neutral localized translation mode.

In the rotating frame of the exact rest orbit, a freely movable localized
carrier must have a neutral collective direction: translating its center by
an infinitesimal amount should be carried into the same translation tangent
by one native update.  This is the discrete analogue of the translational
zero mode of a continuum soliton and uses no Dirac comparator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import canonical_first_shell_phase_embedding as phase


HERE = Path(__file__).resolve().parent


def phase_translation(column: np.ndarray) -> np.ndarray:
    return np.concatenate((column, column))


def classify_dt(dt: float) -> dict:
    phase.DT = dt
    monodromy, metadata = phase.canonical_rotating_monodromy()
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(phase.SIDE, amplitude_sq)
    translations = rank.native.translation_vectors(psi, chi)

    rows = []
    phase_columns = []
    for axis in range(3):
        vector = phase_translation(translations[:, axis])
        vector = vector / np.linalg.norm(vector)
        image = monodromy @ vector
        multiplier = float(np.dot(vector, image))
        neutral_residual = float(np.linalg.norm(image - vector))
        best_scalar_residual = float(np.linalg.norm(
            image - multiplier * vector))
        rows.append({
            "axis": axis,
            "neutral_update_relative_residual": neutral_residual,
            "best_real_multiplier": multiplier,
            "best_scalar_relative_residual": best_scalar_residual,
            "neutral_residual_over_dt_sq": neutral_residual / dt**2,
        })
        phase_columns.append(vector)

    translation_space, _ = np.linalg.qr(np.column_stack(phase_columns))
    response = monodromy @ translation_space
    retained = translation_space @ (translation_space.T @ response)
    cross = float(np.linalg.norm(response - retained)
                  / max(np.linalg.norm(response), 1e-300))

    # The smallest singular directions of M-I are the native directions
    # closest to neutral.  Their overlap with the physical translation
    # tangents shows whether the missing mode merely needs a larger shell.
    _u, singular, vh = np.linalg.svd(monodromy - np.eye(monodromy.shape[0]),
                                     full_matrices=False)
    neutral_basis = vh[-16:].T
    neutral_overlap = float(np.linalg.norm(
        neutral_basis.T @ translation_space)**2 / 3)
    return {
        "dt": dt,
        "metadata": metadata,
        "axis_rows": rows,
        "translation_subspace_cross_block": cross,
        "translation_subspace_cross_block_over_dt_sq": cross / dt**2,
        "smallest_M_minus_I_singular_values": [
            float(value) for value in singular[-16:]],
        "translation_overlap_with_16_nearest_neutral_directions": (
            neutral_overlap),
        "zero_mode_pass": (
            max(row["neutral_update_relative_residual"] for row in rows)
            < 1e-8 and cross < 1e-8),
    }


def main() -> None:
    rank.lock_inertia()
    rows = [classify_dt(1.0 / rate) for rate in (128, 256, 512, 1024)]
    finest = rows[-1]
    normalized = np.asarray([
        row["translation_subspace_cross_block_over_dt_sq"] for row in rows])
    converged_pinning = float(abs(normalized[-1] - normalized[-2])
                              / max(abs(normalized[-1]), 1e-300)) < 0.05
    result = {
        "status": "V4_CANONICAL_TRANSLATION_MODE_OBSTRUCTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "native_action": "canonical V40 position action and centered update",
        "timestep_rows": rows,
        "dt_sq_normalized_cross_block_converged": converged_pinning,
        "verdict": (
            "CANONICAL_TRANSLATION_ZERO_MODE_PASS"
            if finest["zero_mode_pass"] else
            "CANONICAL_TRANSLATION_ZERO_MODE_ABSENT"),
        "claim_boundary": (
            "Linear rotating-frame zero-mode test at the exact localized "
            "rest orbit. Absence proves pinning of this orbit under the "
            "canonical tangent, not a no-go for every deformed action or "
            "finite-amplitude traveling branch."),
    }
    output = HERE / "canonical_translation_mode_obstruction.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
