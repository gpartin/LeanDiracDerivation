"""Screen vacuum-activated native chi-orientation continuity.

The previous uniform chi-dyad stiffness sees the exterior hinge but also
pulls across carrier/vacuum bonds.  This candidate multiplies the same C19
positive square by a native vacuum indicator at both endpoints,

  a_i = ((rho_chi,i-rho_Psi,i)/chi0^2)^2,
  U = eta/(4 chi0^2) sum_(i,n) w_n a_i a_(i+n)
      ||P_chi,i-P_chi,(i+n)||_F^2,
  P_chi = chi chi^dagger.

On the retained carrier rho_chi=rho_Psi and a=0.  In ordinary vacuum Psi=0,
rho_chi=chi0^2 and a=1.  Thus the term controls exterior orientation while
its energy and first force vanish exactly on the carrier/rest configuration.
This is a research action-basis screen, not an admission or promotion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as shift_invert


HERE = Path(__file__).resolve().parent
CACHE = HERE/"personal_touch_vacuum_activated_chi_orientation_cache.npz"
FACE_COEFFICIENTS = np.asarray((
    0.14705479982595762,
    0.14705479982595762,
    0.075,
    0.075,
    0.10294520017404238,
))
STRENGTHS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=-1))


def frobenius_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=(-2, -1)))


def dyad(field: torch.Tensor) -> torch.Tensor:
    return field[..., :, None]*torch.conj(field[..., None, :])


def vacuum_activity(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, cone.SIDE)
    rho_psi = squared_norm(psi)
    rho_chi = squared_norm(chi)
    return ((rho_chi-rho_psi)/small.CHI0**2)**2


def activated_energy(raw: torch.Tensor) -> torch.Tensor:
    _psi, chi = box.unpack(raw, cone.SIDE)
    activity = vacuum_activity(raw)
    orientation = dyad(chi)
    value = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        neighbor_activity = torch.roll(
            activity, shifts=offset, dims=(0, 1, 2))
        neighbor_orientation = torch.roll(
            orientation, shifts=offset, dims=(0, 1, 2))
        difference = orientation-neighbor_orientation
        value += weight*torch.sum(
            activity*neighbor_activity*frobenius_sq(difference))
    return value/(4*small.CHI0**2)


def derive_hessian(state: torch.Tensor) -> np.ndarray:
    hessian = torch.autograd.functional.hessian(
        activated_energy, state, vectorize=True).detach().numpy()
    hessian = (hessian+hessian.T)/2
    np.savez_compressed(CACHE, hessian=hessian)
    return hessian


def main() -> None:
    data = np.load(cone.CACHE)
    rest = torch.tensor(data["rest"], dtype=torch.float64)
    energy = float(activated_energy(rest).detach())
    gradient = torch.func.grad(activated_energy)(rest).detach().numpy()
    hessian = (np.load(CACHE)["hessian"] if CACHE.exists()
               else derive_hessian(rest))
    symmetry = float(np.linalg.norm(hessian-hessian.T)
                     / max(np.linalg.norm(hessian), 1e-300))

    face_hessians = [data[f"hessian_{name}"]
                     for name in shift_invert.NAMES]
    rows = []
    base_stiffness = data["stiffness"]+sum(
        value*hessian for value, hessian in zip(
            FACE_COEFFICIENTS, face_hessians, strict=True))
    for strength in STRENGTHS:
        values = eigh(
            base_stiffness+strength*hessian, data["mass"],
            subset_by_index=[0, 31], eigvals_only=True,
            check_finite=False, driver="gvx")
        rows.append({
            "strength": strength,
            "minimum_generalized_stiffness": float(values[0]),
            "negative_count_in_lowest_32": int(np.sum(values < -1e-8)),
            "lowest_generalized_stiffness": values[:8].tolist(),
        })
        print(strength, rows[-1]["minimum_generalized_stiffness"],
              rows[-1]["negative_count_in_lowest_32"], flush=True)

    best = max(rows, key=lambda row: row["minimum_generalized_stiffness"])
    best_matrix = base_stiffness+best["strength"]*hessian
    residual_values, residual_vectors = eigh(
        best_matrix, data["mass"], subset_by_index=[0, 31],
        check_finite=False, driver="gvx")
    residual_mode = residual_vectors[:, 0]
    shaped = residual_mode.reshape(cone.SIDE, cone.SIDE, cone.SIDE, 4, 3)
    psi_power = float(np.sum(shaped[..., :2, :]**2))
    chi_power = float(np.sum(shaped[..., 2:, :]**2))
    rest_shaped = data["rest"].reshape(
        cone.SIDE, cone.SIDE, cone.SIDE, 4, 3)
    carrier = np.sum(rest_shaped[..., :2, :]**2, axis=(-2, -1)) > 1e-12
    carrier_power = float(np.sum(shaped[carrier]**2))
    total_power = float(np.sum(shaped**2))
    remaining_mode = {
        "generalized_stiffness": float(residual_values[0]),
        "vacuum_activation_curvature": float(
            residual_mode@hessian@residual_mode),
        "Psi_fraction": psi_power/(psi_power+chi_power),
        "chi_fraction": chi_power/(psi_power+chi_power),
        "carrier_fraction": carrier_power/total_power,
        "exterior_fraction": 1-carrier_power/total_power,
    }
    exact = None
    if best["minimum_generalized_stiffness"] >= -1e-8:
        exact_values = shift_invert.spectrum(
            data["mass"], data["gyro"], data["stiffness"],
            face_hessians+[hessian],
            np.r_[FACE_COEFFICIENTS, best["strength"]])
        moduli = np.abs(exact_values)
        exact = {
            "strength": best["strength"],
            "leading_modulus": float(moduli[0]),
            "growth_rate": float(np.log(moduli[0])/small.DT),
            "count_above_1_plus_1e_minus_7_in_32": int(
                np.sum(moduli > 1+1e-7)),
        }
    result = {
        "status": "V4_VACUUM_ACTIVATED_CHI_ORIENTATION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate_action": (
            "sum_C19 w a_i a_j ||Pchi_i-Pchi_j||_F^2/(4 chi0^2), "
            "a=((rhochi-rhoPsi)/chi0^2)^2"),
        "properties": {
            "positive": True,
            "native_fields_only": True,
            "C19_cubic": True,
            "reciprocal": True,
            "phase_invariant": True,
            "contains_prepared_geometry_or_Dirac_data": False,
        },
        "exact_rest_energy": energy,
        "exact_rest_force_norm": float(np.linalg.norm(gradient)),
        "hessian_symmetry_defect": symmetry,
        "face_coefficients": {
            name: float(value) for name, value in zip(
                shift_invert.NAMES, FACE_COEFFICIENTS, strict=True)},
        "rows": rows,
        "best": best,
        "remaining_lowest_mode": remaining_mode,
        "exact_discrete_spectrum_at_positive_stiffness_candidate": exact,
        "verdict": (
            "VACUUM_ACTIVATED_CHI_ORIENTATION_CLOSES_RESOLVED_OUTER_SPECTRUM"
            if exact is not None
            and exact["count_above_1_plus_1e_minus_7_in_32"] == 0 else
            "VACUUM_ACTIVATED_CHI_ORIENTATION_CLOSES_STIFFNESS_GATE"
            if best["minimum_generalized_stiffness"] >= -1e-8 else
            "VACUUM_ACTIVATED_CHI_ORIENTATION_PARTIAL_RESPONSE"),
        "claim_boundary": (
            "Exact-rest Hessian and lowest 32 generalized co-rotating "
            "stiffness values on side five. Exact discrete shift-invert is "
            "run only if the necessary stiffness gate passes. A favorable "
            "row still requires coefficient "
            "provenance, complete-contour spectrum, timestep/box refinement, "
            "Hamiltonian/Noether, nonlinear, and movement gates."),
    }
    output = HERE/"personal_touch_vacuum_activated_chi_orientation.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({"best": best, "verdict": result["verdict"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
