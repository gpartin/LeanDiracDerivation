"""Determine which native joint-field coordinates see every unstable mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import action_response_rank as rank
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def observable(raw: np.ndarray, side: int, kind: str) -> np.ndarray:
    psi, chi = field.prior.bond.local.common.unpack_raw(raw, side)
    if kind == "area":
        value = np.cross(chi, psi)
    elif kind == "overlap":
        value = np.sum(chi.conj() * psi, axis=-1)
    elif kind == "densities":
        value = np.stack((np.sum(abs(psi)**2, axis=-1),
                          np.sum(abs(chi)**2, axis=-1)), axis=-1)
    elif kind == "covariance":
        value = (np.einsum("...a,...b->...ab", psi, psi.conj())
                 + np.einsum("...a,...b->...ab", chi, chi.conj()))
    elif kind == "cross_covariance":
        value = np.einsum("...a,...b->...ab", chi, psi.conj())
    else:
        raise ValueError(kind)
    return np.concatenate((value.real.ravel(), value.imag.ravel()))


def derivative(raw, direction, side, kind, step):
    def real_action(vector):
        return (observable(raw + step * vector, side, kind)
                - observable(raw - step * vector, side, kind)) / (2 * step)
    result = real_action(direction.real)
    if np.linalg.norm(direction.imag) > 0:
        result = result + 1j * real_action(direction.imag)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--growth-threshold", type=float, default=1e-4)
    parser.add_argument("--step", type=float, default=1e-6)
    parser.add_argument("--output", default="unstable_mode_observability.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = rank.exact_rest(args.rate)
    values, vectors = eig(generator, check_finite=False)
    indices = np.where(values.real > args.growth_threshold)[0]
    indices = indices[np.argsort(values[indices].real)[::-1]]
    kinds = ["area", "overlap", "densities", "covariance",
             "cross_covariance"]
    dimension = mass.shape[0]
    rows = []
    for index in indices:
        direction = vectors[:dimension, index]
        qnorm = max(np.linalg.norm(direction), 1e-30)
        psi, chi = field.prior.bond.local.common.unpack_raw(direction, side)
        total = np.sum(abs(psi)**2) + np.sum(abs(chi)**2)
        responses = {
            kind: float(np.linalg.norm(derivative(
                layers[1], direction, side, kind, args.step))
                        / (qnorm * field.prior.R))
            for kind in kinds}
        rows.append({
            "real": float(values[index].real),
            "imag": float(values[index].imag),
            "psi_fraction": float(np.sum(abs(psi)**2) / total),
            "responses": responses,
        })
    summaries = {}
    for kind in kinds:
        data = np.asarray([row["responses"][kind] for row in rows])
        summaries[kind] = {
            "minimum": float(np.min(data)),
            "median": float(np.median(data)),
            "maximum": float(np.max(data)),
            "near_zero_count": int(np.sum(data < 1e-7)),
        }
    result = {
        "status": "V4_UNSTABLE_MODE_OBSERVABILITY",
        "canonical_action_changed": False,
        "rate": args.rate,
        "growth_threshold": args.growth_threshold,
        "mode_count": len(rows),
        "rest_del_rms_residual": residual,
        "summaries": summaries,
        "modes": rows,
        "claim_boundary": (
            "Linear observability only. Seeing a mode does not prove that an "
            "energy built from that observable stabilizes it."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mode_count": len(rows),
        "summaries": summaries,
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
