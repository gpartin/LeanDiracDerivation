"""Test a positive joint Psi--chi bond phase-lock square.

For each unoriented C19 bond define

  C_ij = (chi_i^dagger chi_j) (Psi_i^dagger Psi_j).

Endpoint reversal conjugates C_ij, so Im(C_ij)^2 is reciprocal.  On the
companion-map rest carrier C_ij is real.  The positive square therefore has
zero value and first variation there, while penalizing a relative Psi phase
slip not accompanied by the corresponding chi orientation.  It contains no
prepared sign, corner label, axis, spinor matrix, or target phase.
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
import density_oriented_pair_connection as connection
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    result = raw.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        psij = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        chij = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        psi_overlap = torch.sum(torch.conj(psi)*psij, dim=-1)
        chi_overlap = torch.sum(torch.conj(chi)*chij, dim=-1)
        invariant = torch.imag(chi_overlap*psi_overlap)
        result = result + 0.5*weight*torch.sum(invariant**2)
    # DIRECTIONS contains both orientations.  The factor 1/2 above converts
    # the directed sum to the unordered-bond convention.
    return result/native_time.CHI0**4


def main() -> None:
    bare = np.load(canonical.CACHE)
    rest = torch.tensor(bare["rest"], dtype=torch.float64,
                        requires_grad=True)
    value = potential(rest)
    force = torch.autograd.grad(value, rest, create_graph=True)[0]
    hessian = torch.autograd.functional.hessian(
        potential, rest, vectorize=True).detach().numpy()
    hessian = (hessian+hessian.T)/2

    residual_mode = np.load(
        HERE/"density_connection_fastest_residual_mode.npz")["current"]
    mass0 = np.load(
        HERE/"density_connection_fastest_residual_mode.npz")["mass"]
    denominator = float(np.vdot(residual_mode, mass0@residual_mode).real)
    curvature = float(
        np.vdot(residual_mode, hessian@residual_mode).real/denominator)

    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    cached = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    common = {kind: cached[f"{kind}_common_pair_connection"]
              for kind in ("mass", "gyro", "stiffness")}
    natural = native_time.B*native_time.CHI0
    mass = bare["mass"]+natural*common["mass"]
    gyro = bare["gyro"]+natural*common["gyro"]
    base_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"]
        + natural*common["stiffness"])
    rows = []
    for coefficient in (1/native_time.CHI0, 1.0, native_time.CHI0):
        stiffness = base_stiffness+coefficient*hessian
        dressed = current.dressed_configuration(stiffness, mass, bare["rest"])
        row = spectrum.spectrum_row(
            mass, gyro, stiffness, dressed, native_time.DT)
        row.update({
            "phase_lock_coefficient": coefficient,
            "minimum_mass_eigenvalue": float(eigvalsh(
                mass, subset_by_index=[0, 0], driver="evr")[0]),
        })
        rows.append(row)

    result = {
        "status": "V4_JOINT_PHASE_LOCK_ACTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "sum_C19 w Im[(chi_i^dagger chi_j)"
            "(Psi_i^dagger Psi_j)]^2/(2 chi0^4)"),
        "positive_square": True,
        "rest_value": float(value.detach()),
        "rest_force_norm": float(torch.linalg.vector_norm(force).detach()),
        "unit_curvature_on_previous_residual": curvature,
        "coefficient_scan": rows,
        "verdict": (
            "JOINT_PHASE_LOCK_FULL_SPECTRUM_PASS"
            if any(row["growing_multiplier_count_at_1e_minus_7"] == 0
                   for row in rows)
            else "JOINT_PHASE_LOCK_RETAINS_GROWTH"),
        "claim_boundary": (
            "This is a positive reciprocal research action screen at three "
            "declared geometric normalizations. A linear spectrum pass would "
            "still require coefficient provenance, timestep/box refinement, "
            "nonlinear conservation, movement, and same-state Dirac current."),
    }
    np.savez_compressed(HERE/"joint_phase_lock_action_hessian.npz",
                        hessian=hessian)
    (HERE/"joint_phase_lock_action_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
