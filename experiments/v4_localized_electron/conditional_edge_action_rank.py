"""Response-rank audit for the cubical conditional edge comparator.

This is the signed-current counterpart of ``edge_snap_action_rank.py``.
The cubical projectors are used only to prepare an independent comparator;
they never enter an action column or the native force.  The purpose of the
probe is to determine whether action bases retired against a rigid carrier
shift acquire the missing response once the four components are transported
in their derived forward/backward edge channels.

The zero-phase control is deliberately massless.  It isolates the odd
transport channel before any local beta/mass mixing is added.  Passing this
rank test would be necessary, not sufficient, for a massive localized Dirac
branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import conditional_edge_native_gate as conditional
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5


def conditional_layers(axis: int, sign: int, dt: float,
                       psi: np.ndarray, chi: np.ndarray):
    del dt  # The endpoint map is geometric; dt enters the DEL residual.
    previous, following, _ = conditional.target(
        (0.0, 0.0), psi, chi, axis, sign)
    return previous, (psi, chi), following


def assemble(names, rows, psi, chi, force_psi, force_chi):
    matrices = []
    targets = []
    direction_rows = []
    for row in rows:
        axis = int(row["axis"])
        sign = int(row["direction"])
        dt = float(row["best_dt"])
        layers = conditional_layers(axis, sign, dt, psi, chi)
        residual = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        matrix = np.column_stack([
            rank.action_delta(*layers, dt, name) for name in names
        ])
        norm = np.linalg.norm(residual)
        matrices.append(matrix / norm)
        targets.append(-residual / norm)
        correlations = []
        for column in range(matrix.shape[1]):
            column_norm = np.linalg.norm(matrix[:, column])
            correlations.append(float(
                np.vdot(matrix[:, column], -residual).real
                / max(column_norm * norm, 1e-30)))
        direction_rows.append({
            "axis": axis,
            "sign": sign,
            "canonical_del_residual_norm": float(norm),
            "basis_rank": int(np.linalg.matrix_rank(matrix, tol=1e-11)),
            "maximum_absolute_single_column_correlation": float(
                max(abs(value) for value in correlations)),
        })
    return matrices, targets, direction_rows


def parity(matrices, targets, rows):
    converted = [
        {"axis": int(row["axis"]), "sign": int(row["direction"])}
        for row in rows
    ]
    return rank.parity_ratios(matrices, targets, converted)


def run_basis(names, rows, psi, chi, force_psi, force_chi):
    matrices, targets, direction_rows = assemble(
        names, rows, psi, chi, force_psi, force_chi)
    return {
        "direction_rows": direction_rows,
        "direction_reversal_parity": parity(matrices, targets, rows),
        "unconstrained_fit": rank.fit_span(names, matrices, targets),
        "nonnegative_fit": rank.fit_span(
            names, matrices, targets, nonnegative=True),
        "best_single": rank.best_single(names, matrices, targets),
    }


def main() -> None:
    rank.lock_inertia()
    gate = json.loads((HERE / "conditional_edge_native_gate.json").read_text(
        encoding="utf-8"))
    rows = gate["rows"]
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)

    gyroscopic = run_basis(
        rank.GYROSCOPIC, rows, psi, chi, force_psi, force_chi)
    symmetric = run_basis(
        rank.SYMMETRIC_KINETIC, rows, psi, chi, force_psi, force_chi)
    result = {
        "status": "V4_CONDITIONAL_EDGE_ACTION_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "comparator": (
            "zero-phase cubical conditional edge shift; projectors appear "
            "only in comparator preparation"),
        "gyroscopic": gyroscopic,
        "symmetric_kinetic": symmetric,
        "verdict": "RESPONSE_RANK_CLASSIFICATION_ONLY",
        "claim_boundary": (
            "This is a necessary massless signed-channel response test. It "
            "does not prove a lossless physical-corner spinor embedding, a "
            "massive beta step, rest stability, or free carrier motion."),
    }
    (HERE / "conditional_edge_action_rank.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
