"""Show why a rigid one-edge translation is not the massive Dirac target.

The matrices are reconstructed from the already-proved cubical incidence
algebra. Dirac is used only as an independent comparator. A Strang quantum
walk conditionally shifts the +/- gamma channels to adjacent lattice sites
and mixes them locally with beta. The whole eight-component carrier does not
undergo one common shift except in a massless one-chirality control.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import cubical_edge_shift_channels as cube


HERE = Path(__file__).resolve().parent


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def main() -> None:
    identity = np.eye(8, dtype=complex)
    alpha = cube.gamma(0)
    beta = cube.beta()
    flavor = cube.flavor()
    copy_plus = (identity + flavor) / 2
    chiral_plus = (identity + alpha) / 2
    chiral_minus = (identity - alpha) / 2

    mass = 1.0
    momentum = 0.6
    energy = float(np.sqrt(mass**2 + momentum**2))
    hamiltonian = momentum * alpha + mass * beta
    energy_plus = (identity + hamiltonian / energy) / 2
    seed = np.asarray([1, 2j, -1j, 0.5, -0.7, 0.3j, 1.2, -0.4j])
    state = normalized(energy_plus @ copy_plus @ seed)

    plus_weight = float(np.vdot(
        chiral_plus @ state, chiral_plus @ state).real)
    minus_weight = float(np.vdot(
        chiral_minus @ state, chiral_minus @ state).real)
    expected_plus = (1 + momentum / energy) / 2
    expected_minus = (1 - momentum / energy) / 2

    rows = []
    for dt in (0.2, 0.1, 0.05, 0.025):
        k = momentum * dt
        exact = expm(-1j * dt * hamiltonian)
        spatial = (np.exp(-1j * k) * chiral_plus
                   + np.exp(1j * k) * chiral_minus)
        half_mass = expm(-0.5j * dt * mass * beta)
        walk = half_mass @ spatial @ half_mass
        rigid = np.exp(-1j * k) * identity
        rows.append({
            "dt": dt,
            "conditional_walk_state_error": float(np.linalg.norm(
                (walk-exact) @ state)),
            "conditional_walk_generator_error": float(np.linalg.norm(
                ((walk-identity)/dt + 1j*hamiltonian) @ state)),
            "rigid_shift_state_error": float(np.linalg.norm(
                (rigid-exact) @ state)),
            "rigid_shift_generator_error": float(np.linalg.norm(
                ((rigid-identity)/dt + 1j*hamiltonian) @ state)),
            "conditional_walk_flavor_leakage": float(np.linalg.norm(
                (identity-copy_plus) @ walk @ state)),
        })

    massless_forward = normalized(chiral_plus @ copy_plus @ seed)
    k = 0.371
    conditional_massless = (
        np.exp(-1j*k)*chiral_plus + np.exp(1j*k)*chiral_minus)
    rigid_forward = np.exp(-1j*k)*identity
    massless_control_error = float(np.linalg.norm(
        (conditional_massless-rigid_forward) @ massless_forward))

    result = {
        "status": "V4_RIGID_SHIFT_VS_DIRAC_TRANSPORT_DIAGNOSIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mass": mass,
        "momentum": momentum,
        "energy": energy,
        "dirac_group_velocity": momentum/energy,
        "positive_edge_channel_weight": plus_weight,
        "negative_edge_channel_weight": minus_weight,
        "expected_positive_weight": expected_plus,
        "expected_negative_weight": expected_minus,
        "channel_weight_defect": max(
            abs(plus_weight-expected_plus),
            abs(minus_weight-expected_minus)),
        "rows": rows,
        "massless_single_channel_rigid_shift_error": massless_control_error,
        "issue": (
            "The prior edge gate translated the complete massive carrier in "
            "one direction as one scalar object."),
        "why_it_matters": (
            "Massive Dirac transport conditionally sends two internal "
            "channels to each adjacent edge and mixes them locally; a rigid "
            "whole-carrier shift is not the equation being derived."),
        "symptoms": (
            "The conditional walk converges to the cubical Dirac generator "
            "under timestep refinement, while the rigid-shift generator "
            "retains an order-one mismatch. Rigid shift is exact only for the "
            "massless single-chirality control."),
        "expected_symptoms": (
            "A correct native movement bridge should reproduce signed +/- "
            "gamma bond currents and beta mixing inside each flavor copy."),
        "root_cause": (
            "The test target conflated carrier-envelope translation with the "
            "microscopic conditional motion of its four spinor channels."),
        "corrective_action": (
            "Replace the rigid endpoint gate with a frozen cubical readout "
            "that measures the two signed edge-channel currents and local "
            "mass mixing. Then compare their recombination to Dirac current; "
            "do not install the conditional shift in the native action."),
        "copy_result": (
            "Both flavor copies obey the same conditional shift independently; "
            "transport does not exchange or alternate the copies."),
        "claim_boundary": (
            "This repairs the independent comparator. It does not establish "
            "that the current Psi-chi action generates these channel currents."),
    }
    (HERE / "rigid_shift_vs_dirac_transport.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
