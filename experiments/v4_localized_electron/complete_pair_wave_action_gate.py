"""Gate the native complete-pair C19 wave action.

Let Z_i = chi_i tensor Psi_i.  The canonical area field chi wedge Psi is only
the antisymmetric part of this pair.  The action below gives the complete pair
the same native C19 bond wave balance at the already-derived rest frequency
chi0:

  Delta L = eta/(2 chi0^4) sum_<ij> w [|Zdot_j-Zdot_i|^2
                                      -chi0^2 |Z_j-Z_i|^2].

It is real, local, reciprocal, separately global-phase invariant, cubic/C19
symmetric, and introduces no direction, corner label, spinor, or target track.
The unit coefficient is screened only; no action is admitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eig, eigvalsh, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import current_action_localized_envelope_closure as envelope
import current_action_two_layer_spectrum as current
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5


def pair(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    return chi[..., :, None]*psi[..., None, :]


def bond_norm(value: torch.Tensor) -> torch.Tensor:
    result = value.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        difference = value-torch.roll(
            value, shifts=offset, dims=(0, 1, 2))
        result = result + weight*torch.sum(
            torch.real(torch.conj(difference)*difference))
    return result


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    zl, zr = pair(left), pair(right)
    kinetic = bond_norm((zr-zl)/dt)
    potential = (bond_norm(zl)+bond_norm(zr))/2
    return (dt/(2*native_time.CHI0**4)
            * (kinetic-native_time.CHI0**2*potential))


def derive_blocks(rest: np.ndarray, dt: float):
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    current_layer = torch.tensor(
        rest, dtype=torch.float64, requires_grad=True)
    previous = torch.tensor(
        rotation.T@rest, dtype=torch.float64, requires_grad=True)
    following = torch.tensor(
        rotation@rest, dtype=torch.float64, requires_grad=True)

    def residual(left, middle, right):
        action = interval(left, middle, dt)+interval(middle, right, dt)
        return torch.autograd.grad(action, middle, create_graph=True)[0]

    rest_residual = residual(previous, current_layer, following).detach().numpy()
    blocks = []
    for argument in range(3):
        def function(value, selected=argument):
            layers = [previous, current_layer, following]
            layers[selected] = value
            return residual(*layers)
        blocks.append(torch.autograd.functional.jacobian(
            function, (previous, current_layer, following)[argument],
            vectorize=True).detach().numpy())
    minus = blocks[0]@rotation.T
    zero = blocks[1]
    plus = blocks[2]@rotation
    mass = -0.5*dt*(plus+minus)
    gyro = -(plus-minus)
    stiffness = -(plus+zero+minus)/dt
    return {
        "mass": (mass+mass.T)/2,
        "gyro": (gyro-gyro.T)/2,
        "stiffness": (stiffness+stiffness.T)/2,
        "rest_residual": rest_residual,
    }


def spectrum_row(mass: np.ndarray, gyro: np.ndarray, stiffness: np.ndarray,
                 dressed: np.ndarray, dt: float) -> dict:
    factor = lu_factor(mass/dt+gyro/2)
    forward = -lu_solve(
        factor, (dt*stiffness-2*mass/dt)@dressed)
    backward = -lu_solve(factor, (mass/dt-gyro/2)@dressed)
    residual = np.column_stack((
        forward-dressed@(dressed.T@forward),
        backward-dressed@(dressed.T@backward)))
    u, singular, _ = np.linalg.svd(residual, full_matrices=False)
    extra = u[:, singular > 1e-10*max(float(singular[0]), 1.0)]
    hull = envelope.orth(np.column_stack((dressed, extra)))
    fh = -lu_solve(factor, (dt*stiffness-2*mass/dt)@hull)
    bh = -lu_solve(factor, (mass/dt-gyro/2)@hull)
    f, g = hull.T@fh, hull.T@bh
    identity = np.eye(hull.shape[1])
    monodromy = np.block([[f, g], [identity, np.zeros_like(identity)]])
    values = eig(monodromy, right=False, check_finite=False)
    modulus = abs(values)
    closure = np.column_stack((fh-hull@f, bh-hull@g))
    return {
        "configuration_hull_rank": int(hull.shape[1]),
        "phase_space_dimension": int(monodromy.shape[0]),
        "hull_relative_cross_block": float(
            np.linalg.norm(closure)/np.linalg.norm(np.column_stack((fh, bh)))),
        "maximum_multiplier_modulus": float(max(modulus)),
        "maximum_growth_rate": float(np.log(max(modulus))/dt),
        "growing_multiplier_count_at_1e_minus_7": int(np.sum(
            modulus > 1+1e-7)),
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    delta = derive_blocks(bare["rest"], native_time.DT)
    base_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    mode = np.load(HERE/"current_action_fastest_mode.npz")["current"]
    old_denominator = float(np.vdot(mode, bare["mass"]@mode).real)
    rows = []
    for coefficient in (1.0, native_time.CHI0, native_time.CHI0**2):
        mass = bare["mass"]+coefficient*delta["mass"]
        gyro = bare["gyro"]+coefficient*delta["gyro"]
        stiffness = base_stiffness+coefficient*delta["stiffness"]
        row = spectrum_row(
            mass, gyro, stiffness,
            current.dressed_configuration(stiffness, mass, bare["rest"]),
            native_time.DT)
        row["coefficient"] = coefficient
        row["mass_minimum_eigenvalue"] = float(eigvalsh(
            mass, subset_by_index=[0, 0], driver="evr")[0])
        rows.append(row)
    result = {
        "status": "V4_COMPLETE_PAIR_WAVE_ACTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate": {
            "field": "Z_i = chi_i tensor Psi_i",
            "coefficients_screened": [1.0, native_time.CHI0,
                                      native_time.CHI0**2],
            "kinetic_potential_frequency": native_time.CHI0,
            "C19_only": True,
            "positive_Hamiltonian_contribution": True,
            "contains_Dirac_or_prepared_track_data": False,
        },
        "rest_DEL_norm": float(np.linalg.norm(delta["rest_residual"])),
        "old_unstable_mode_contributions": {
            "added_mass": float(
                np.vdot(mode, delta["mass"]@mode).real/old_denominator),
            "added_gyro_magnitude": float(
                abs(np.vdot(mode, delta["gyro"]@mode))/old_denominator),
            "added_stiffness": float(
                np.vdot(mode, delta["stiffness"]@mode).real/old_denominator),
            "old_stiffness": -1.0807423181893132,
        },
        "two_layer_geometric_coefficient_scan": rows,
        "verdict": (
            "PAIR_WAVE_UNIT_CANDIDATE_PASSES_LINEAR_REST_GATE"
            if (np.linalg.norm(delta["rest_residual"]) < 1e-8
                and any(row["growing_multiplier_count_at_1e_minus_7"] == 0
                        for row in rows))
            else "PAIR_WAVE_GEOMETRIC_SCAN_RETAINS_LINEAR_GROWTH"),
        "claim_boundary": (
            "Exact action variation and complete finite-hull linear spectrum "
            "at the three declared geometric coefficients. A pass would still "
            "require coefficient "
            "provenance, dt/box refinement, nonlinear conservation, motion, "
            "and same-state Dirac current. No action is installed here."),
    }
    (HERE/"complete_pair_wave_action_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
