"""Isolate the native C19 chi-orientation action on the limiting hinge.

This is a response and branch-solvability diagnostic, not an action
admission.  The candidate contains only the native complex chi field and the
canonical C19 bonds:

  U_dir = (1/(4 chi0^2)) sum_(i,n) w_n
          ||chi_i chi_i^dagger-chi_(i+n) chi_(i+n)^dagger||_F^2.

It is positive, cubic, reciprocal, locally phase invariant, and zero in a
uniform vacuum.  Unlike W=chi wedge Psi it remains sensitive to chi direction
where Psi vanishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs
from torch.func import grad, jvp

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native
import personal_touch_face_psd_cone as cone


HERE = Path(__file__).resolve().parent
DT = 1/256
SIGMA = 1.000002
UNIFORM_ADDITION = 10.0
NAMES = tuple(name for name, _kind, _part in cone.FACE_BASIS)


def dyad(field: torch.Tensor) -> torch.Tensor:
    return field[..., :, None]*torch.conj(field[..., None, :])


def frobenius_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=(-2, -1)))


def chi_orientation_energy(raw: torch.Tensor) -> torch.Tensor:
    _psi, chi = box.unpack(raw, cone.SIDE)
    direction = dyad(chi)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        delta = direction-torch.roll(
            direction, shifts=offset, dims=(0, 1, 2))
        result += weight*torch.sum(frobenius_sq(delta))
    return result/(4*small.CHI0**2)


def leading_mode(data, coefficients: np.ndarray):
    mass, gyro, stiffness = data["mass"], data["gyro"], data["stiffness"]
    hessians = [data[f"hessian_{name}"] for name in NAMES]
    candidate = stiffness+sum(
        value*hessian for value, hessian in zip(
            coefficients, hessians, strict=True))
    left = -mass/DT-gyro/2
    backward = -mass/DT+gyro/2
    zero = 2*mass/DT-DT*candidate
    left_factor = lu_factor(left)
    qfactor = lu_factor(SIGMA**2*left+SIGMA*zero+backward)
    dimension = len(mass)

    def matvec(raw):
        current, previous = raw[:dimension], raw[dimension:]
        following = -lu_solve(
            left_factor, zero@current+backward@previous)
        return np.r_[following, current]

    def inverse(raw):
        upper, lower = raw[:dimension], raw[dimension:]
        previous = -lu_solve(
            qfactor, left@upper+(zero+SIGMA*left)@lower)
        return np.r_[lower+SIGMA*previous, previous]

    operator = LinearOperator(
        (2*dimension, 2*dimension), matvec=matvec, dtype=float)
    opinverse = LinearOperator(
        (2*dimension, 2*dimension), matvec=inverse, dtype=float)
    values, vectors = eigs(
        operator, k=8, sigma=SIGMA, OPinv=opinverse, which="LM",
        tol=1e-11, maxiter=4000, ncv=48)
    index = int(np.argmax(np.abs(values)))
    return values[index], vectors[:dimension, index]


def main() -> None:
    data = np.load(cone.CACHE)
    base = np.asarray((
        0.14705479982595762,
        0.14705479982595762,
        0.075,
        0.075,
        0.10294520017404238,
    ))
    coefficients = base+UNIFORM_ADDITION
    eigenvalue, mode = leading_mode(data, coefficients)
    state = torch.tensor(data["rest"], dtype=torch.float64)
    function = chi_orientation_energy
    energy = float(function(state).detach())
    gradient = grad(function)(state)
    images = []
    curvature = 0.0
    for direction in (np.real(mode), np.imag(mode)):
        tangent = torch.tensor(direction, dtype=torch.float64)
        _value, image = jvp(grad(function), (state,), (tangent,))
        image_np = image.detach().numpy()
        images.append(image_np)
        curvature += float(np.dot(direction, image_np))
    norm_sq = float(np.vdot(mode, mode).real)
    image_norm = float(np.sqrt(sum(np.dot(value, value) for value in images)))

    vacuum = np.zeros_like(data["rest"])
    _psi, chi = box.unpack(torch.tensor(vacuum), cone.SIDE)
    with torch.no_grad():
        chi[..., 2] = small.CHI0
    vacuum_energy = float(function(torch.tensor(vacuum)).detach())

    shaped = mode.reshape(cone.SIDE, cone.SIDE, cone.SIDE, 4, 3)
    psi_power = float(np.sum(abs(shaped[..., :2, :])**2))
    chi_power = float(np.sum(abs(shaped[..., 2:, :])**2))
    result = {
        "status": "V4_CHI_ORIENTATION_LIMITING_HINGE_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "sum_C19 w ||chi_i chi_i^dagger-"
            "chi_j chi_j^dagger||_F^2/(4 chi0^2)"),
        "properties": {
            "positive": True,
            "reciprocal": True,
            "cubic_C19": True,
            "local_chi_phase_invariant": True,
            "uniform_vacuum_energy": vacuum_energy,
        },
        "reference_face_uniform_addition": UNIFORM_ADDITION,
        "limiting_eigenvalue": {
            "real": float(eigenvalue.real),
            "imag": float(eigenvalue.imag),
            "modulus": float(abs(eigenvalue)),
        },
        "energy_at_exact_rest": energy,
        "rest_force_norm": float(torch.linalg.vector_norm(gradient).detach()),
        "limiting_mode_curvature": curvature/norm_sq,
        "limiting_mode_image_norm_per_mode_norm": image_norm/np.sqrt(norm_sq),
        "limiting_mode_field_power": {
            "Psi_fraction": psi_power/(psi_power+chi_power),
            "chi_fraction": chi_power/(psi_power+chi_power),
        },
        "verdict": (
            "CHI_ORIENTATION_ACTION_SEES_LIMITING_HINGE_AND_REQUIRES_RESOLVED_REST"
            if curvature > 0 and float(torch.linalg.vector_norm(gradient)) > 1e-10
            else "CHI_ORIENTATION_ACTION_DOES_NOT_SUPPLY_REQUIRED_RESPONSE"),
        "claim_boundary": (
            "Hessian response at the exact retained rest orbit. Nonzero rest "
            "force requires a freely re-solved branch before spectral use."),
    }
    output = HERE/"personal_touch_chi_orientation_hinge.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
