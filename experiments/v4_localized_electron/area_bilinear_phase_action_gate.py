"""Screen the Z2 phase stiffness of the native area field.

The native area W_i=chi_i wedge Psi_i already appears in V40.  The complex
bilinear omega_i=W_i dot W_i transforms by one common global phase.  Hence
Im(conj(omega_i) omega_j)^2 is invariant under the declared independent
global Psi and chi phases, reciprocal on an unordered bond, cubic, and
positive.  The current rest carrier has omega_i real with the two allowed Z2
signs, so this square vanishes without naming either sign or a corner.
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
    area = torch.linalg.cross(chi, psi, dim=-1)
    omega = torch.sum(area*area, dim=-1)
    result = raw.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        omegaj = torch.roll(omega, shifts=offset, dims=(0, 1, 2))
        phase_area = torch.imag(torch.conj(omega)*omegaj)
        result = result + 0.5*weight*torch.sum(phase_area**2)
    return result


def main() -> None:
    bare = np.load(canonical.CACHE)
    rest = torch.tensor(bare["rest"], dtype=torch.float64,
                        requires_grad=True)
    raw_value = raw_potential(rest)
    raw_force = torch.autograd.grad(raw_value, rest, create_graph=True)[0]
    raw_hessian = torch.autograd.functional.hessian(
        raw_potential, rest, vectorize=True).detach().numpy()
    raw_hessian = (raw_hessian+raw_hessian.T)/2
    normalized_hessian = raw_hessian/native_time.CHI0**12

    mode_cache = np.load(HERE/"density_connection_fastest_residual_mode.npz")
    mode, mass = mode_cache["current"], mode_cache["mass"]
    denominator = float(np.vdot(mode, mass@mode).real)
    curvature = float(
        np.vdot(mode, normalized_hessian@mode).real/denominator)

    mass, gyro, base_stiffness = (
        mode_cache["mass"], mode_cache["gyro"], mode_cache["stiffness"])
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
        "status": "V4_AREA_BILINEAR_PHASE_ACTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "sum_C19 w Im[conj(omega_i) omega_j]^2/(2 chi0^12), "
            "omega_i=(chi_i wedge Psi_i) dot (chi_i wedge Psi_i)"),
        "positive_square": True,
        "preserves_global_U1xU1": True,
        "normalized_rest_value": float(
            raw_value.detach()/native_time.CHI0**12),
        "normalized_rest_force_norm": float(
            torch.linalg.vector_norm(raw_force).detach()
            / native_time.CHI0**12),
        "normalized_curvature_on_residual": curvature,
        "coefficient_scan": rows,
        "verdict": (
            "AREA_PHASE_ACTION_FULL_SPECTRUM_PASS"
            if any(row["growing_multiplier_count_at_1e_minus_7"] == 0
                   for row in rows)
            else "AREA_PHASE_ACTION_RETAINS_GROWTH"),
        "claim_boundary": (
            "The phase square is a research action screen, not an installed "
            "term. A linear pass would still require coefficient provenance, "
            "refinements, nonlinear conservation, motion, and current."),
    }
    np.savez_compressed(HERE/"area_bilinear_phase_action_hessian.npz",
                        normalized_hessian=normalized_hessian)
    (HERE/"area_bilinear_phase_action_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
