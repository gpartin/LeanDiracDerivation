"""Leading side-five Floquet spectrum of the completed transfer oscillator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_oscillator_implicit_update as update


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def rotation_matrix(dimension: int, theta: float) -> np.ndarray:
    matrix = np.eye(dimension)
    shaped = np.arange(dimension).reshape(SIDE, SIDE, SIDE, 4, 3)
    c, s = np.cos(theta), np.sin(theta)
    real = shaped[..., 0, :].ravel()
    imag = shaped[..., 1, :].ravel()
    matrix[real, real] = c
    matrix[real, imag] = -s
    matrix[imag, real] = s
    matrix[imag, imag] = c
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--even-strength", type=float, default=0.0)
    parser.add_argument("--parallel-strength", type=float, default=0.0)
    parser.add_argument("--density-balance-strength", type=float, default=0.0)
    parser.add_argument("--spectral-area-strength", type=float, default=0.0)
    parser.add_argument("--loop-curvature-strength", type=float, default=0.0)
    parser.add_argument("--face-factorization-strength", type=float, default=0.0)
    parser.add_argument("--face-conjugate-strength", type=float, default=0.0)
    parser.add_argument("--face-edge-commutator-strength", type=float, default=0.0)
    parser.add_argument("--dt-divisor", type=int, default=1)
    parser.add_argument("--eigenvalues", type=int, default=32)
    parser.add_argument("--classify-leading", action="store_true")
    parser.add_argument("--save-leading-mode", action="store_true")
    parser.add_argument("--eigen-tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--ncv", type=int, default=None)
    args = parser.parse_args()
    if args.dt_divisor < 1:
        raise ValueError("--dt-divisor must be positive")
    small.DT /= args.dt_divisor
    update.STRENGTH = args.strength
    update.EVEN_STRENGTH = args.even_strength
    update.PARALLEL_STRENGTH = args.parallel_strength
    update.DENSITY_BALANCE_STRENGTH = args.density_balance_strength
    update.SPECTRAL_AREA_STRENGTH = args.spectral_area_strength
    update.LOOP_CURVATURE_STRENGTH = args.loop_curvature_strength
    update.FACE_FACTORIZATION_STRENGTH = args.face_factorization_strength
    update.FACE_CONJUGATE_FACTORIZATION_STRENGTH = args.face_conjugate_strength
    update.FACE_EDGE_COMMUTATOR_STRENGTH = args.face_edge_commutator_strength
    rest = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    current = torch.tensor(rest, dtype=torch.float64)
    previous = box.phase_rotate(current, np.conjugate(phase), SIDE).detach()
    following = box.phase_rotate(current, phase, SIDE).detach()
    residual = update.del_residual(previous, current, following).detach().numpy()
    blocks = []
    for argument in range(3):
        def function(value, index=argument):
            layers = [previous, current, following]
            layers[index] = value
            return update.del_residual(*layers)
        point = (previous, current, following)[argument]
        blocks.append(torch.autograd.functional.jacobian(
            function, point, vectorize=True).detach().numpy())
    a, b, cblock = blocks
    rotation = rotation_matrix(rest.size, theta)
    lhs = cblock@rotation
    backward_block = a@rotation.T
    singular = np.linalg.svd(lhs, compute_uv=False)
    factor = lu_factor(lhs)
    dimension = rest.size

    def matvec(raw):
        current_part = raw[:dimension]
        previous_part = raw[dimension:]
        following_part = -lu_solve(
            factor, b@current_part+backward_block@previous_part)
        return np.r_[following_part, current_part]

    operator = LinearOperator((2*dimension, 2*dimension), matvec=matvec,
                              dtype=np.float64)
    rng = np.random.default_rng(4011)
    initial = rng.normal(size=2*dimension)
    try:
        eigenvalues, eigenvectors = eigs(
            operator, k=args.eigenvalues, which="LM", v0=initial,
            tol=args.eigen_tol, maxiter=args.maxiter, ncv=args.ncv,
            return_eigenvectors=True)
        converged = True
    except ArpackNoConvergence as error:
        eigenvalues = error.eigenvalues
        eigenvectors = error.eigenvectors
        converged = False
        if eigenvalues is None or len(eigenvalues) == 0:
            raise
    moduli = np.sort(np.abs(eigenvalues))[::-1]
    leading_mode = None
    leading_vector = None
    leading_value = None
    if args.classify_leading:
        leading_index = int(np.argmax(np.abs(eigenvalues)))
        vector = eigenvectors[:, leading_index]
        leading_vector = vector
        leading_value = eigenvalues[leading_index]
        current_mode = vector[:dimension].reshape(SIDE, SIDE, SIDE, 4, 3)
        core = np.zeros((SIDE, SIDE, SIDE), dtype=bool)
        core[1:3, 1:3, 1:3] = True
        psi_power = np.sum(np.abs(current_mode[..., :2, :])**2, axis=(-2, -1))
        chi_power = np.sum(np.abs(current_mode[..., 2:, :])**2, axis=(-2, -1))
        total_power = float(np.sum(psi_power)+np.sum(chi_power))
        flat_mode = current_mode.reshape(-1)
        mode_power = float(np.vdot(flat_mode, flat_mode).real)
        rest_shaped = rest.reshape(SIDE, SIDE, SIDE, 4, 3)

        def subspace_fraction(columns: list[np.ndarray]) -> float:
            matrix = np.column_stack(columns)
            q, _ = np.linalg.qr(matrix)
            return float(np.sum(np.abs(q.T@flat_mode)**2)/mode_power)

        translations = []
        for axis in range(3):
            translations.append((
                np.roll(rest_shaped, 1, axis=axis)
                - np.roll(rest_shaped, -1, axis=axis)
            ).reshape(-1)/2)
        psi_phase = np.zeros_like(rest_shaped)
        psi_phase[..., 0, :] = -rest_shaped[..., 1, :]
        psi_phase[..., 1, :] = rest_shaped[..., 0, :]
        chi_phase = np.zeros_like(rest_shaped)
        chi_phase[..., 2, :] = -rest_shaped[..., 3, :]
        chi_phase[..., 3, :] = rest_shaped[..., 2, :]
        psi_amplitude = np.zeros_like(rest_shaped)
        psi_amplitude[..., :2, :] = rest_shaped[..., :2, :]
        vacuum = np.zeros_like(rest_shaped)
        vacuum[..., 2, 2] = small.CHI0
        chi_contrast_amplitude = np.zeros_like(rest_shaped)
        chi_contrast_amplitude[..., 2:, :] = (
            rest_shaped[..., 2:, :]-vacuum[..., 2:, :])
        site_rows = []
        for x in range(SIDE):
            for y in range(SIDE):
                for z in range(SIDE):
                    power = float(psi_power[x, y, z]+chi_power[x, y, z])
                    site_rows.append({
                        "site": [x, y, z],
                        "total_power_fraction": power/total_power,
                        "psi_power_fraction": float(
                            psi_power[x, y, z]/total_power),
                        "chi_power_fraction": float(
                            chi_power[x, y, z]/total_power),
                        "is_carrier_corner": bool(core[x, y, z]),
                    })
        site_rows.sort(key=lambda row: row["total_power_fraction"],
                       reverse=True)
        leading_mode = {
            "eigenvalue_real": float(np.real(eigenvalues[leading_index])),
            "eigenvalue_imag": float(np.imag(eigenvalues[leading_index])),
            "modulus": float(np.abs(eigenvalues[leading_index])),
            "psi_power_fraction": float(np.sum(psi_power)/total_power),
            "chi_power_fraction": float(np.sum(chi_power)/total_power),
            "carrier_cube_power_fraction": float(
                (np.sum(psi_power[core])+np.sum(chi_power[core]))/total_power),
            "exterior_power_fraction": float(
                (np.sum(psi_power[~core])+np.sum(chi_power[~core]))/total_power),
            "carrier_psi_power_fraction": float(np.sum(psi_power[core])/total_power),
            "carrier_chi_power_fraction": float(np.sum(chi_power[core])/total_power),
            "exterior_psi_power_fraction": float(np.sum(psi_power[~core])/total_power),
            "exterior_chi_power_fraction": float(np.sum(chi_power[~core])/total_power),
            "translation_subspace_power_fraction": subspace_fraction(translations),
            "psi_phase_power_fraction": subspace_fraction([psi_phase.reshape(-1)]),
            "chi_phase_power_fraction": subspace_fraction([chi_phase.reshape(-1)]),
            "psi_amplitude_power_fraction": subspace_fraction([psi_amplitude.reshape(-1)]),
            "chi_contrast_amplitude_power_fraction": subspace_fraction(
                [chi_contrast_amplitude.reshape(-1)]),
            "largest_site_participation": site_rows[:16],
        }
    result = {
        "status": "V4_TRANSFER_OSCILLATOR_LEADING_SPECTRUM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "strength": args.strength,
        "even_strength": args.even_strength,
        "parallel_strength": args.parallel_strength,
        "density_balance_strength": args.density_balance_strength,
        "spectral_area_strength": args.spectral_area_strength,
        "loop_curvature_strength": args.loop_curvature_strength,
        "face_factorization_strength": args.face_factorization_strength,
        "face_conjugate_strength": args.face_conjugate_strength,
        "face_edge_commutator_strength": args.face_edge_commutator_strength,
        "dt": small.DT,
        "dt_divisor": args.dt_divisor,
        "side": SIDE,
        "phase_space_dimension": int(2*dimension),
        "rest_DEL_rms": float(np.linalg.norm(residual)/np.sqrt(residual.size)),
        "following_block_condition_number": float(
            singular[0]/singular[-1]),
        "computed_eigenvalue_count": int(moduli.size),
        "arpack_converged": converged,
        "leading_moduli": moduli.tolist(),
        "leading_spectral_radius": float(moduli[0]),
        "leading_growth_rate": float(np.log(moduli[0])/small.DT),
        "count_above_1_plus_1e_minus_7": int(np.sum(moduli > 1+1e-7)),
        "count_above_1_plus_1e_minus_6": int(np.sum(moduli > 1+1e-6)),
        "verdict": (
            "NO_EXPONENTIAL_GROWTH_IN_LEADING_COMPUTED_SPECTRUM"
            if moduli[0] < 1+1e-7 else
            "LEADING_SPECTRUM_HAS_EXPONENTIAL_GROWTH"),
        "leading_mode_classification": leading_mode,
        "claim_boundary": (
            f"{moduli.size} converged largest-modulus Ritz values of the complete side-five "
            "two-layer linearization. A favorable result is not a full dense "
            "spectrum or nonlinear long-time stability proof."),
    }
    suffix = f"{args.strength:g}".replace("-", "m").replace(".", "p")
    if args.even_strength != 0:
        even_suffix = f"{args.even_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_even_{even_suffix}"
    if args.parallel_strength != 0:
        parallel_suffix = f"{args.parallel_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_parallel_{parallel_suffix}"
    if args.density_balance_strength != 0:
        density_suffix = f"{args.density_balance_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_density_{density_suffix}"
    if args.spectral_area_strength != 0:
        area_suffix = f"{args.spectral_area_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_spectral_area_{area_suffix}"
    if args.loop_curvature_strength != 0:
        loop_suffix = f"{args.loop_curvature_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_loop_curvature_{loop_suffix}"
    if args.face_factorization_strength != 0:
        face_suffix = f"{args.face_factorization_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_face_factorization_{face_suffix}"
    if args.face_conjugate_strength != 0:
        conjugate_suffix = f"{args.face_conjugate_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_face_conjugate_{conjugate_suffix}"
    if args.face_edge_commutator_strength != 0:
        edge_suffix = f"{args.face_edge_commutator_strength:g}".replace("-", "m").replace(".", "p")
        suffix += f"_face_edge_commutator_{edge_suffix}"
    if args.dt_divisor != 1:
        suffix += f"_dt_div_{args.dt_divisor}"
    output = HERE / f"canonical_transfer_oscillator_spectrum_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    if args.save_leading_mode:
        if leading_vector is None:
            raise ValueError("--save-leading-mode requires --classify-leading")
        np.savez_compressed(
            HERE / f"canonical_transfer_oscillator_spectrum_{suffix}_leading_mode.npz",
            eigenvalue=np.asarray(leading_value),
            phase_space_vector=leading_vector,
            rest=rest,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
