"""Linear rest-stability screen for one declared V4 action.

This is a cheap gate before a nonlinear 100-cycle run.  It consumes the three
exact discrete Euler--Lagrange tangent blocks of the declared interval action,
forms the full co-rotating two-layer update, and asks for its largest Floquet
multipliers.  A multiplier outside the unit circle blocks V4-002; a unit-circle
screen is necessary but still requires the nonlinear gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-module", required=True)
    parser.add_argument("--action-description", required=True)
    parser.add_argument("--cache-provenance", default=(
        "Exact tangent cache inherited from checkpoint d56b63a; the recorded "
        "SHA-256 is the authoritative identity."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eigenvalues", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=12000)
    parser.add_argument("--ncv", type=int, default=96)
    parser.add_argument("--solver", choices=("arnoldi", "dense"),
                        default="arnoldi")
    args = parser.parse_args()

    cache = args.cache.resolve()
    if not cache.is_file():
        raise FileNotFoundError(cache)
    action_module = HERE / args.action_module
    if not action_module.is_file():
        raise FileNotFoundError(action_module)
    data = np.load(cache)
    previous, current, following = (
        data[name] for name in ("previous", "current", "following"))
    if previous.shape != current.shape or current.shape != following.shape:
        raise ValueError("incompatible tangent block shapes")

    dimension = previous.shape[0]
    if dimension % 12 != 0:
        raise ValueError("tangent cache is not a native Psi/chi lattice")
    side = round((dimension / 12) ** (1 / 3))
    if side**3 * 12 != dimension:
        raise ValueError("tangent cache is not a cubic native lattice")
    theta = float(np.arccos(
        1 - native_time.DT**2 * native_time.CHI0**2 / 2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    lhs = following @ rotate_plus
    backward = previous @ rotate_minus
    factor = lu_factor(lhs, check_finite=False)

    def matvec(raw: np.ndarray) -> np.ndarray:
        q = raw[:dimension]
        p = raw[dimension:]
        successor = -lu_solve(
            factor, current @ q + backward @ p, check_finite=False)
        return np.concatenate((successor, q))

    if args.solver == "dense":
        forward = -lu_solve(factor, current, check_finite=False)
        history = -lu_solve(factor, backward, check_finite=False)
        identity = np.eye(dimension)
        zero = np.zeros_like(identity)
        monodromy = np.block([[forward, history], [identity, zero]])
        values, vectors = eig(monodromy, check_finite=False)
        converged = True
    else:
        operator = LinearOperator(
            (2 * dimension, 2 * dimension), matvec=matvec,
            dtype=np.float64)
        rng = np.random.default_rng(4002)
        try:
            values, vectors = eigs(
                operator, k=args.eigenvalues, which="LM",
                v0=rng.normal(size=2 * dimension), tol=args.tolerance,
                maxiter=args.maxiter, ncv=args.ncv,
                return_eigenvectors=True)
            converged = True
        except ArpackNoConvergence as error:
            values, vectors = error.eigenvalues, error.eigenvectors
            converged = False
            if values is None or len(values) == 0:
                raise
    moduli = np.abs(values)
    leading = int(np.argmax(moduli))
    current_mode = vectors[:dimension, leading].reshape(
        side, side, side, 4, 3)
    power = np.sum(np.abs(current_mode)**2, axis=-1)
    total = max(float(np.sum(power)), 1e-300)
    result = {
        "stage": "V4-002",
        "status": "LINEAR_STABILITY_SCREEN",
        "action_module": args.action_module,
        "action_module_sha256": sha256(action_module),
        "action_description": args.action_description,
        "cache_file": cache.name,
        "cache_provenance": args.cache_provenance,
        "cache_sha256": sha256(cache),
        "dt": native_time.DT,
        "state_dimension": dimension,
        "box_side": side,
        "phase_space_dimension": 2 * dimension,
        "eigensolver": args.solver,
        "computed_eigenvalue_count": int(len(values)),
        "eigensolver_converged": converged,
        "leading_multiplier_real": float(values[leading].real),
        "leading_multiplier_imag": float(values[leading].imag),
        "leading_multiplier_modulus": float(moduli[leading]),
        "leading_growth_rate_per_unit_time": float(
            np.log(moduli[leading]) / native_time.DT),
        "leading_mode_field_power_fractions": {
            "Psi": float(np.sum(power[..., :2]) / total),
            "chi": float(np.sum(power[..., 2:]) / total),
        },
        "verdict": (
            "LINEAR_STABILITY_NECESSARY_GATE_PASS"
            if converged and float(moduli[leading]) <= 1 + 1e-7
            else "LINEAR_STABILITY_FAIL"),
        "claim_boundary": (
            "This is a full two-layer tangent screen, not the nonlinear "
            "100-cycle V4-002 acceptance gate."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
