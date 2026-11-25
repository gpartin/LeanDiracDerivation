"""Cutting-plane positive cone for the five zero-on-rest face Hessians.

This calculation replaces repeated nonnormal Floquet coefficient scans by a
variational stability question.  At the exact rotating carrier it derives the
co-rotating mass, gyroscopic and stiffness matrices from the three-layer DEL
operator.  It then asks whether a nonnegative combination of the independent
positive face Hessians makes the stiffness nonnegative on every direction.

Positive stiffness with positive mass is a sufficient energy criterion for
spectral stability of the continuous co-rotating gyroscopic linearization.
It is not by itself the finite-dt Floquet or nonlinear movement gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import linprog

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_transfer_oscillator_implicit_update as update
import canonical_transfer_oscillator_spectrum as spectrum
import personal_touch_face_identity_basis as face


HERE = Path(__file__).resolve().parent
SIDE = face.SIDE
CACHE = HERE/"personal_touch_face_psd_cone_cache.npz"


FACE_BASIS = (
    ("diagonal_symmetric", "diagonal", "symmetric"),
    ("diagonal_antisymmetric", "diagonal", "antisymmetric"),
    ("diagonal_reversed", "diagonal_reversed", "full"),
    ("diagonal_conjugate", "diagonal_conjugate", "full"),
    ("edge_commutator", "edge_commutator", "full"),
)


def derive_matrices() -> dict[str, np.ndarray]:
    update.STRENGTH = 0.0
    update.EVEN_STRENGTH = 2.0
    update.PARALLEL_STRENGTH = 0.0
    update.DENSITY_BALANCE_STRENGTH = 0.5
    update.SPECTRAL_AREA_STRENGTH = 0.0
    update.LOOP_CURVATURE_STRENGTH = 0.0
    update.FACE_FACTORIZATION_STRENGTH = 0.0
    update.FACE_CONJUGATE_FACTORIZATION_STRENGTH = 0.0
    update.FACE_EDGE_COMMUTATOR_STRENGTH = 0.0

    rest = np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    current = torch.tensor(rest, dtype=torch.float64)
    previous = box.phase_rotate(current, np.conjugate(phase), SIDE).detach()
    following = box.phase_rotate(current, phase, SIDE).detach()
    blocks = []
    for argument in range(3):
        def function(value, index=argument):
            layers = [previous, current, following]
            layers[index] = value
            return update.del_residual(*layers)
        blocks.append(torch.autograd.functional.jacobian(
            function, (previous, current, following)[argument],
            vectorize=True).detach().numpy())
    rotation = spectrum.rotation_matrix(rest.size, theta)
    minus, zero, plus = blocks[0]@rotation.T, blocks[1], blocks[2]@rotation
    mass = -0.5*small.DT*(plus+minus)
    gyro = -(plus-minus)
    stiffness = -(plus+zero+minus)/small.DT
    matrices: dict[str, np.ndarray] = {
        "rest": rest,
        "mass": (mass+mass.T)/2,
        "gyro": (gyro-gyro.T)/2,
        "stiffness": (stiffness+stiffness.T)/2,
    }
    for name, kind, part in FACE_BASIS:
        function = lambda raw, k=kind, p=part: face.energy(raw, k, p)
        hessian = torch.autograd.functional.hessian(
            function, current, vectorize=True).detach().numpy()
        matrices[f"hessian_{name}"] = (hessian+hessian.T)/2
    np.savez(CACHE, **matrices)
    return matrices


def main() -> None:
    matrices = ({key: value for key, value in np.load(CACHE).items()}
                if CACHE.exists() else derive_matrices())
    mass = matrices["mass"]
    stiffness = matrices["stiffness"]
    hessians = [matrices[f"hessian_{name}"] for name, _, _ in FACE_BASIS]
    dimension = mass.shape[0]
    mass_values = eigh(mass, subset_by_index=[0, min(7, dimension-1)],
                       eigvals_only=True, check_finite=False)
    if mass_values[0] <= 0:
        raise RuntimeError("co-rotating mass is not positive definite")

    constraints_h: list[np.ndarray] = []
    constraints_k: list[float] = []
    coefficients = np.zeros(len(hessians))
    trace = []
    tolerance = 1e-8
    for iteration in range(12):
        candidate = stiffness+sum(value*hessian for value, hessian
                                  in zip(coefficients, hessians, strict=True))
        count = min(63, dimension-1)
        values, vectors = eigh(
            candidate, mass, subset_by_index=[0, count],
            check_finite=False, driver="gvx")
        negative = np.where(values < -tolerance)[0]
        trace.append({
            "iteration": iteration,
            "minimum_generalized_stiffness": float(values[0]),
            "negative_count_in_lowest_block": int(negative.size),
            "coefficients": coefficients.tolist(),
        })
        if negative.size == 0:
            break
        for index in negative:
            vector = vectors[:, index]
            constraints_k.append(float(vector@stiffness@vector))
            constraints_h.append(np.asarray(
                [vector@hessian@vector for hessian in hessians], dtype=float))
        hmatrix = np.vstack(constraints_h)
        kvals = np.asarray(constraints_k)
        # k + H c >= 0  <=>  -H c <= k.
        solution = linprog(
            np.ones(len(hessians)), A_ub=-hmatrix, b_ub=kvals,
            bounds=[(0.0, None)]*len(hessians), method="highs")
        if not solution.success:
            trace[-1]["linear_program_failure"] = solution.message
            break
        coefficients = solution.x

    final = stiffness+sum(value*hessian for value, hessian
                          in zip(coefficients, hessians, strict=True))
    final_values = eigh(
        final, mass, subset_by_index=[0, min(127, dimension-1)],
        eigvals_only=True, check_finite=False, driver="gvx")
    passed = bool(final_values[0] >= -tolerance)
    result = {
        "status": "V4_FACE_PSD_CONE_DERIVATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "dimension": dimension,
        "mass_lowest_eigenvalues": mass_values.tolist(),
        "basis": [name for name, _, _ in FACE_BASIS],
        "trace": trace,
        "final_coefficients": {
            name: float(value)
            for (name, _, _), value in zip(FACE_BASIS, coefficients, strict=True)
            if value > 1e-12
        },
        "final_lowest_generalized_stiffness": final_values[:32].tolist(),
        "negative_count_in_final_lowest_block": int(
            np.sum(final_values < -tolerance)),
        "verdict": (
            "POSITIVE_FACE_CONE_CLOSES_CONTINUOUS_ENERGY_INDEX"
            if passed else "POSITIVE_FACE_CONE_REMAINS_INDEFINITE"),
        "claim_boundary": (
            "Positive-mass continuous co-rotating energy criterion at the "
            "exact side-five rest orbit. A pass still requires exact finite-"
            "dt Floquet, timestep/box refinement, nonlinear stability, "
            "coefficient provenance, movement, and Dirac-current gates."),
    }
    (HERE/"personal_touch_face_psd_cone.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
