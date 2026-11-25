"""Complete native bond-class orientation basis at the exact carrier.

Native density selectors distinguish bonds without site labels:

  a_i=((rho_chi-rho_Psi)/chi0^2)^2  (vacuum selector),
  c_i=(rho_Psi/chi0^2)^2            (carrier selector).

The three positive C19 energies vanish with zero first force on their exact
bond classes: aligned vacuum-vacuum chi, orthogonal carrier-vacuum chi, and
the C3 tight-frame carrier-carrier overlap |<chi_i,chi_j>|^2=rho_i rho_j/3.
They are screened as an action basis only; no coefficient is admitted.
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
import localized_stability as native
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan


HERE = Path(__file__).resolve().parent
CACHE = HERE/"personal_touch_activated_bond_basis_v2_cache.npz"
NAMES = (
    "vacuum_vacuum_chi",
    "carrier_vacuum_chi",
    "carrier_carrier_chi",
    "carrier_carrier_Psi_tight_frame",
)
FACE_COEFFICIENTS = np.asarray((
    0.14705479982595762, 0.14705479982595762,
    0.075, 0.075, 0.10294520017404238))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=-1))


def fields_and_selectors(raw: torch.Tensor):
    psi, chi = box.unpack(raw, cone.SIDE)
    rho_psi, rho_chi = squared_norm(psi), squared_norm(chi)
    vacuum = ((rho_chi-rho_psi)/small.CHI0**2)**2
    carrier = (rho_psi/small.CHI0**2)**2
    return psi, chi, rho_psi, rho_chi, vacuum, carrier


def energies(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    psi, chi, rho_psi, rho, vacuum, carrier = fields_and_selectors(raw)
    values = [torch.zeros((), dtype=torch.float64) for _ in NAMES]
    for offset, weight in native.DIRECTIONS:
        neighbor = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_psi = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho_psi = torch.roll(
            rho_psi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho = torch.roll(rho, shifts=offset, dims=(0, 1, 2))
        neighbor_vacuum = torch.roll(
            vacuum, shifts=offset, dims=(0, 1, 2))
        neighbor_carrier = torch.roll(
            carrier, shifts=offset, dims=(0, 1, 2))
        overlap = torch.sum(torch.conj(chi)*neighbor, dim=-1)
        s = torch.real(torch.conj(overlap)*overlap)
        p = rho*neighbor_rho
        values[0] += weight*torch.sum(
            vacuum*neighbor_vacuum*(p-s))/(4*small.CHI0**2)
        interface = vacuum*neighbor_carrier+carrier*neighbor_vacuum
        values[1] += weight*torch.sum(interface*s)/(4*small.CHI0**2)
        values[2] += weight*torch.sum(
            carrier*neighbor_carrier*(3*s-p)**2)/(4*small.CHI0**6)
        psi_overlap = torch.sum(
            torch.conj(psi)*neighbor_psi, dim=-1)
        psi_s = torch.real(torch.conj(psi_overlap)*psi_overlap)
        psi_p = rho_psi*neighbor_rho_psi
        values[3] += weight*torch.sum(
            (3*psi_s-psi_p)**2)/(4*small.CHI0**6)
    return tuple(values)


def derive(state: torch.Tensor) -> dict[str, np.ndarray]:
    result = {}
    for index, name in enumerate(NAMES):
        function = lambda raw, selected=index: energies(raw)[selected]
        hessian = torch.autograd.functional.hessian(
            function, state, vectorize=True).detach().numpy()
        result[f"hessian_{name}"] = (hessian+hessian.T)/2
    np.savez_compressed(CACHE, **result)
    return result


def main() -> None:
    data = np.load(cone.CACHE)
    state = torch.tensor(data["rest"], dtype=torch.float64)
    rest_energies = [float(value.detach()) for value in energies(state)]
    rest_force_norms = []
    for index in range(len(NAMES)):
        function = lambda raw, selected=index: energies(raw)[selected]
        rest_force_norms.append(float(torch.linalg.vector_norm(
            torch.func.grad(function)(state)).detach()))
    cached = ({key: value for key, value in np.load(CACHE).items()}
              if CACHE.exists() else derive(state))
    hessians = [cached[f"hessian_{name}"] for name in NAMES]
    face_hessians = [data[f"hessian_{name}"] for name in face_scan.NAMES]
    base = data["stiffness"]+sum(
        value*hessian for value, hessian in zip(
            FACE_COEFFICIENTS, face_hessians, strict=True))

    coefficients = np.zeros(len(hessians))
    constraint_h, constraint_k, trace = [], [], []
    for iteration in range(10):
        candidate = base+sum(value*hessian for value, hessian in zip(
            coefficients, hessians, strict=True))
        values, vectors = eigh(
            candidate, data["mass"], subset_by_index=[0, 63],
            check_finite=False, driver="gvx")
        negative = np.where(values < -1e-8)[0]
        trace.append({
            "iteration": iteration,
            "coefficients": coefficients.tolist(),
            "minimum_generalized_stiffness": float(values[0]),
            "negative_count_in_lowest_64": int(len(negative)),
        })
        if len(negative) == 0:
            break
        for index in negative:
            vector = vectors[:, index]
            constraint_k.append(float(vector@base@vector))
            constraint_h.append(np.asarray(
                [vector@hessian@vector for hessian in hessians]))
        solution = linprog(
            np.ones(len(hessians)), A_ub=-np.vstack(constraint_h),
            b_ub=np.asarray(constraint_k),
            bounds=[(0.0, None)]*len(hessians), method="highs")
        if not solution.success:
            trace[-1]["linear_program_failure"] = solution.message
            break
        coefficients = solution.x

    final = base+sum(value*hessian for value, hessian in zip(
        coefficients, hessians, strict=True))
    final_values = eigh(
        final, data["mass"], subset_by_index=[0, 63],
        eigvals_only=True, check_finite=False, driver="gvx")
    result = {
        "status": "V4_ACTIVATED_BOND_CLASS_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": list(NAMES),
        "properties": {
            "positive": True,
            "native_Psi_chi_only": True,
            "C19_cubic": True,
            "reciprocal": True,
            "phase_invariant": True,
            "contains_site_labels_or_Dirac_data": False,
        },
        "exact_rest_energies": dict(zip(NAMES, rest_energies, strict=True)),
        "exact_rest_force_norms": dict(zip(
            NAMES, rest_force_norms, strict=True)),
        "positive_stiffness_cone_trace": trace,
        "final_coefficients": dict(zip(
            NAMES, coefficients.tolist(), strict=True)),
        "final_lowest_generalized_stiffness": final_values[:16].tolist(),
        "verdict": (
            "ACTIVATED_BOND_BASIS_CLOSES_SUFFICIENT_STIFFNESS_GATE"
            if final_values[0] >= -1e-8 else
            "ACTIVATED_BOND_BASIS_DOES_NOT_CLOSE_SUFFICIENT_STIFFNESS_GATE"),
        "claim_boundary": (
            "Positive exact-rest Hessian cone only. Positive co-rotating "
            "stiffness is sufficient but not necessary for discrete "
            "gyroscopic stability. No coefficient or action is admitted."),
    }
    (HERE/"personal_touch_activated_bond_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
