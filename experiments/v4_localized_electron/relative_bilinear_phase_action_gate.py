"""Screen the minimal U(1)xU(1)-invariant relative-phase square.

The local bilinear

  tau_i = conjugate(Psi_i dot Psi_i) (chi_i dot chi_i)

records twice the relative Psi/chi phase.  A global independent phase change
multiplies every tau_i by the same phase, so tau_i^* tau_j is invariant.  The
positive bond square Im(tau_i^* tau_j)^2 has no selected sign: both aligned
and anti-aligned Z2 branches are minima.  On the current companion carrier
all tau_i are real, so the term and its first variation vanish exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import complete_pair_wave_action_gate as spectrum
import current_action_two_layer_spectrum as current
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def raw_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    sigma_psi = torch.sum(psi*psi, dim=-1)
    sigma_chi = torch.sum(chi*chi, dim=-1)
    tau = torch.conj(sigma_psi)*sigma_chi
    result = raw.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        tauj = torch.roll(tau, shifts=offset, dims=(0, 1, 2))
        phase_area = torch.imag(torch.conj(tau)*tauj)
        result = result + 0.5*weight*torch.sum(phase_area**2)
    return result


def main() -> None:
    bare = np.load(canonical.CACHE)
    rest = torch.tensor(bare["rest"], dtype=torch.float64,
                        requires_grad=True)
    raw_value = raw_potential(rest)
    raw_force = torch.autograd.grad(
        raw_value, rest, create_graph=True)[0]
    raw_hessian = torch.autograd.functional.hessian(
        raw_potential, rest, vectorize=True).detach().numpy()
    raw_hessian = (raw_hessian+raw_hessian.T)/2

    mode_cache = np.load(HERE/"density_connection_fastest_residual_mode.npz")
    mode, mass = mode_cache["current"], mode_cache["mass"]
    denominator = float(np.vdot(mode, mass@mode).real)
    raw_curvature = float(np.vdot(mode, raw_hessian@mode).real/denominator)

    # Degree sixteen divided by chi0^12 has the same quartic field degree as
    # the native local action.  No residual or Dirac velocity enters it.
    normalized_hessian = raw_hessian/native_time.CHI0**12
    normalized_curvature = raw_curvature/native_time.CHI0**12

    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    connection = np.load(HERE/"density_connection_fastest_residual_mode.npz")
    mass, gyro, base_stiffness = (
        connection["mass"], connection["gyro"], connection["stiffness"])
    rows = []
    for coefficient in (1/native_time.CHI0**2,
                        1/native_time.CHI0, 1.0):
        stiffness = base_stiffness+coefficient*normalized_hessian
        dressed = current.dressed_configuration(stiffness, mass, bare["rest"])
        row = spectrum.spectrum_row(
            mass, gyro, stiffness, dressed, native_time.DT)
        row.update({
            "coefficient": coefficient,
            "minimum_mass_eigenvalue": float(eigvalsh(
                mass, subset_by_index=[0, 0], driver="evr")[0]),
        })
        rows.append(row)

    result = {
        "status": "V4_RELATIVE_BILINEAR_PHASE_ACTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "sum_C19 w Im[conj(tau_i) tau_j]^2/(2 chi0^12), "
            "tau_i=conj(Psi_i dot Psi_i)(chi_i dot chi_i)"),
        "positive_square": True,
        "preserves_global_U1xU1": True,
        "normalized_rest_value": float(
            raw_value.detach()/native_time.CHI0**12),
        "normalized_rest_force_norm": float(
            torch.linalg.vector_norm(raw_force).detach()
            / native_time.CHI0**12),
        "raw_curvature_on_residual": raw_curvature,
        "normalized_curvature_on_residual": normalized_curvature,
        "coefficient_scan": rows,
        "verdict": (
            "RELATIVE_PHASE_ACTION_FULL_SPECTRUM_PASS"
            if any(row["growing_multiplier_count_at_1e_minus_7"] == 0
                   for row in rows)
            else "RELATIVE_PHASE_ACTION_RETAINS_GROWTH"),
        "claim_boundary": (
            "The phase square is derived from native bilinears and screened "
            "at three native coefficient scales. A spectral pass would still "
            "require minimality, coefficient provenance, refinements, "
            "nonlinear conservation, motion, and Dirac-current agreement."),
    }
    np.savez_compressed(HERE/"relative_bilinear_phase_action_hessian.npz",
                        raw_hessian=raw_hessian,
                        normalized_hessian=normalized_hessian)
    (HERE/"relative_bilinear_phase_action_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
