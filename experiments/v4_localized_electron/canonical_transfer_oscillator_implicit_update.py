"""Implicit variational update and discrete Noether audit for the candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_oscillator_gate as oscillator


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE
MODE = "transfer_odd"
STRENGTH = 1.0
EVEN_STRENGTH = 0.0
PARALLEL_STRENGTH = 0.0
DENSITY_BALANCE_STRENGTH = 0.0
SPECTRAL_AREA_STRENGTH = 0.0
LOOP_CURVATURE_STRENGTH = 0.0
FACE_FACTORIZATION_STRENGTH = 0.0
FACE_CONJUGATE_FACTORIZATION_STRENGTH = 0.0
FACE_EDGE_COMMUTATOR_STRENGTH = 0.0


def parallel_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    overlap = torch.sum(torch.conj(chi)*psi, dim=-1)
    return torch.sum(torch.real(torch.conj(overlap)*overlap))


def c19_dirichlet(field: torch.Tensor) -> torch.Tensor:
    return sum(weight*(field-torch.roll(
        field, shifts=offset, dims=(0, 1, 2)))
        for offset, weight in oscillator.native.DIRECTIONS)


def density_balance_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi)*chi, dim=-1))
    return torch.sum(rho_psi**2*(rho_chi-rho_psi)**2)/(4*small.CHI0**4)


def spectral_area_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    defect = c19_dirichlet(c19_dirichlet(area)-27*area)
    norm = torch.real(torch.sum(torch.conj(defect)*defect, dim=-1))
    return torch.sum(norm)/(small.CHI0**2*27**2)


def loop_curvature_energy(raw: torch.Tensor) -> torch.Tensor:
    """Cubic sum of full-frame plaquette commutator squares.

    The native Hermitian frame is P=Psi Psi^dagger+chi chi^dagger.  The
    commutator of its two one-bond changes measures noncommuting/torsional
    response around a square face.  The sum has no selected direction.  The
    chi0^4 denominator reduces the degree-eight numerator to the same field
    dimension as the native quartic potential; the remaining coefficient is
    dimensionless and is not selected here.
    """
    psi, chi = box.unpack(raw, SIDE)
    frame = (psi[..., :, None]*torch.conj(psi[..., None, :])
             + chi[..., :, None]*torch.conj(chi[..., None, :]))
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        shift_first = tuple(1 if axis == first else 0 for axis in range(3))
        shift_second = tuple(1 if axis == second else 0 for axis in range(3))
        da = torch.roll(frame, shifts=shift_first, dims=(0, 1, 2))-frame
        db = torch.roll(frame, shifts=shift_second, dims=(0, 1, 2))-frame
        curvature = da@db-db@da
        result += torch.sum(torch.real(torch.conj(curvature)*curvature))/2
    return result/small.CHI0**4


def face_factorization_energy(raw: torch.Tensor) -> torch.Tensor:
    """Positive square-face consistency defect of the native oriented area.

    With W=chi wedge Psi and w=W/chi0, each elementary face compares the
    tensor product on one diagonal with the tensor product on the other.  All
    three face planes are summed, so no direction, corner, or copy is chosen.
    """
    psi, chi = box.unpack(raw, SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)/small.CHI0
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        sf = tuple(1 if axis == first else 0 for axis in range(3))
        ss = tuple(1 if axis == second else 0 for axis in range(3))
        wa = torch.roll(w, shifts=sf, dims=(0, 1, 2))
        wb = torch.roll(w, shifts=ss, dims=(0, 1, 2))
        wab = torch.roll(w, shifts=tuple(sf[axis]+ss[axis]
                         for axis in range(3)), dims=(0, 1, 2))
        defect = (w[..., :, None]*wab[..., None, :]
                  - wa[..., :, None]*wb[..., None, :])
        result += torch.sum(torch.real(torch.conj(defect)*defect))/2
    return result


def face_conjugate_factorization_energy(raw: torch.Tensor) -> torch.Tensor:
    """Independent phase-neutral diagonal factorization of square faces."""
    psi, chi = box.unpack(raw, SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)/small.CHI0
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        sf = tuple(1 if axis == first else 0 for axis in range(3))
        ss = tuple(1 if axis == second else 0 for axis in range(3))
        wa = torch.roll(w, shifts=sf, dims=(0, 1, 2))
        wb = torch.roll(w, shifts=ss, dims=(0, 1, 2))
        wab = torch.roll(w, shifts=tuple(sf[axis]+ss[axis]
                         for axis in range(3)), dims=(0, 1, 2))
        defect = (w[..., :, None]*torch.conj(wab[..., None, :])
                  - wa[..., :, None]*torch.conj(wb[..., None, :]))
        result += torch.sum(torch.real(torch.conj(defect)*defect))/2
    return result


def face_edge_commutator_energy(raw: torch.Tensor) -> torch.Tensor:
    """Positive noncommuting-edge response on every square face."""
    psi, chi = box.unpack(raw, SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)/small.CHI0
    result = torch.zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        sf = tuple(1 if axis == first else 0 for axis in range(3))
        ss = tuple(1 if axis == second else 0 for axis in range(3))
        da = torch.roll(w, shifts=sf, dims=(0, 1, 2))-w
        db = torch.roll(w, shifts=ss, dims=(0, 1, 2))-w
        defect = (da[..., :, None]*db[..., None, :]
                  - db[..., :, None]*da[..., None, :])
        result += torch.sum(torch.real(torch.conj(defect)*defect))/2
    return result


def interval(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (box.interval(left, right, SIDE)
            + STRENGTH*oscillator.oscillator_interval(left, right, MODE)
            + EVEN_STRENGTH*oscillator.oscillator_interval(
                left, right, "transfer_even")
            - PARALLEL_STRENGTH*small.DT/2*(
                parallel_energy(left)+parallel_energy(right))
            - DENSITY_BALANCE_STRENGTH*small.DT/2*(
                density_balance_energy(left)+density_balance_energy(right))
            - SPECTRAL_AREA_STRENGTH*small.DT/2*(
                spectral_area_energy(left)+spectral_area_energy(right))
            - LOOP_CURVATURE_STRENGTH*small.DT/2*(
                loop_curvature_energy(left)+loop_curvature_energy(right))
            - FACE_FACTORIZATION_STRENGTH*small.DT/2*(
                face_factorization_energy(left)
                + face_factorization_energy(right))
            - FACE_CONJUGATE_FACTORIZATION_STRENGTH*small.DT/2*(
                face_conjugate_factorization_energy(left)
                + face_conjugate_factorization_energy(right))
            - FACE_EDGE_COMMUTATOR_STRENGTH*small.DT/2*(
                face_edge_commutator_energy(left)
                + face_edge_commutator_energy(right)))


del_residual = grad(
    lambda p, c, f: interval(p, c)+interval(c, f), argnums=1)
right_momentum = grad(interval, argnums=1)


def generator(raw: torch.Tensor, field: str) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    zp, zc = torch.zeros_like(psi), torch.zeros_like(chi)
    return torch.stack((
        torch.real(1j*psi if field == "Psi" else zp),
        torch.imag(1j*psi if field == "Psi" else zp),
        torch.real(1j*chi if field == "chi" else zc),
        torch.imag(1j*chi if field == "chi" else zc)),
        dim=-2).reshape(-1)


def charge(left: torch.Tensor, right: torch.Tensor, field: str) -> float:
    return float(torch.dot(
        right_momentum(left, right), generator(right, field)).detach())


def fixed_newton(previous: np.ndarray, current: np.ndarray,
                 guess: np.ndarray, factor, argument: str) -> tuple[np.ndarray, list]:
    state = guess.copy()
    trace = []
    for iteration in range(12):
        tensors = [torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, state)]
        if argument == "following":
            value = del_residual(*tensors).detach().numpy()
        else:
            value = del_residual(tensors[2], tensors[1], tensors[0]) \
                .detach().numpy()
        rms = np.linalg.norm(value)/np.sqrt(value.size)
        trace.append({"iteration": iteration, "residual_rms": float(rms)})
        if rms < 1e-11:
            break
        correction = lu_solve(factor, -value)
        accepted = False
        for power in range(12):
            trial = state+0.5**power*correction
            tt = torch.tensor(trial, dtype=torch.float64)
            if argument == "following":
                trial_value = del_residual(tensors[0], tensors[1], tt)
            else:
                trial_value = del_residual(tt, tensors[1], tensors[0])
            trial_rms = float(torch.linalg.vector_norm(trial_value).detach()
                              / np.sqrt(value.size))
            if trial_rms < rms:
                state = trial
                trace[-1]["accepted_scale"] = float(0.5**power)
                accepted = True
                break
        if not accepted:
            trace[-1]["accepted_scale"] = 0.0
            break
    return state, trace


def main() -> None:
    rest = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    current = torch.tensor(rest, dtype=torch.float64)
    previous = box.phase_rotate(current, np.conjugate(phase), SIDE).detach()
    following = box.phase_rotate(current, phase, SIDE).detach()
    rest_residual = del_residual(previous, current, following).detach().numpy()

    block_following = torch.autograd.functional.jacobian(
        lambda value: del_residual(previous, current, value), following,
        vectorize=True).detach().numpy()
    block_previous = torch.autograd.functional.jacobian(
        lambda value: del_residual(value, current, following), previous,
        vectorize=True).detach().numpy()
    factor_following = lu_factor(block_following)
    factor_previous = lu_factor(block_previous)
    rng = np.random.default_rng(4007)
    raw_direction = rng.normal(size=rest.size)
    raw_direction /= np.linalg.norm(raw_direction)
    rest_norm = np.linalg.norm(rest)
    rows = []
    for epsilon in (1e-7, 1e-6, 1e-5, 1e-4):
        dprevious = epsilon*rest_norm*raw_direction
        # Exact linear compatible prediction for the following layer.
        image = block_previous@dprevious
        dfollowing = lu_solve(factor_following, -image)
        p = previous.detach().numpy()+dprevious
        c = current.detach().numpy()
        guess = following.detach().numpy()+dfollowing
        solved, forward_trace = fixed_newton(
            p, c, guess, factor_following, "following")
        recovered, reverse_trace = fixed_newton(
            solved, c, p, factor_previous, "previous")
        pt, ct, ft = [torch.tensor(value, dtype=torch.float64)
                      for value in (p, c, solved)]
        qpsi_left = charge(pt, ct, "Psi")
        qpsi_right = charge(ct, ft, "Psi")
        qchi_left = charge(pt, ct, "chi")
        qchi_right = charge(ct, ft, "chi")
        rows.append({
            "relative_previous_perturbation": epsilon,
            "forward_final_residual_rms": forward_trace[-1]["residual_rms"],
            "forward_iterations": len(forward_trace),
            "reverse_final_residual_rms": reverse_trace[-1]["residual_rms"],
            "reverse_iterations": len(reverse_trace),
            "reversibility_relative_error": float(
                np.linalg.norm(recovered-p)/max(np.linalg.norm(p), 1e-300)),
            "Psi_charge_left": qpsi_left,
            "Psi_charge_right": qpsi_right,
            "Psi_charge_relative_error": float(
                abs(qpsi_right-qpsi_left)/max(abs(qpsi_left), 1e-300)),
            "chi_charge_left": qchi_left,
            "chi_charge_right": qchi_right,
            "chi_charge_absolute_error": float(abs(qchi_right-qchi_left)),
            "chi_charge_relative_error": float(
                abs(qchi_right-qchi_left)/max(abs(qchi_left), 1.0)),
            "nonlinear_correction_over_linear_prediction": float(
                np.linalg.norm(solved-guess)
                / max(np.linalg.norm(dfollowing), 1e-300)),
        })
    passed = all(row["forward_final_residual_rms"] < 1e-10
                 and row["reverse_final_residual_rms"] < 1e-10
                 and row["reversibility_relative_error"] < 1e-10
                 and row["Psi_charge_relative_error"] < 1e-10
                 and row["chi_charge_relative_error"] < 1e-10
                 for row in rows)
    singular = np.linalg.svd(block_following, compute_uv=False)
    result = {
        "status": "V4_TRANSFER_OSCILLATOR_IMPLICIT_UPDATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mode": MODE,
        "strength": STRENGTH,
        "even_strength": EVEN_STRENGTH,
        "parallel_strength": PARALLEL_STRENGTH,
        "density_balance_strength": DENSITY_BALANCE_STRENGTH,
        "spectral_area_strength": SPECTRAL_AREA_STRENGTH,
        "rest_DEL_rms": float(
            np.linalg.norm(rest_residual)/np.sqrt(rest_residual.size)),
        "following_Legendre_block_condition_number": float(
            singular[0]/singular[-1]),
        "rows": rows,
        "verdict": (
            "IMPLICIT_UPDATE_REVERSIBLE_AND_NOETHER_CONSERVING"
            if passed else "IMPLICIT_UPDATE_GATE_FAIL"),
        "claim_boundary": (
            "Local nonlinear perturbations around the exact rest orbit. A pass "
            "does not establish long-time stability, spatial-current equality, "
            "or finite electron translation."),
    }
    output = HERE / "canonical_transfer_oscillator_implicit_update.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
