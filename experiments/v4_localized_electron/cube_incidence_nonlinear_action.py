"""One nonlinear action for the unit cube-incidence research candidate.

This module reconstructs every term whose tangent matrices entered the
successful linear spectrum.  It audits directional derivatives against those
assembled matrices before the action is used by an implicit updater.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import area_bilinear_phase_action_gate as area_phase
import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_geometry_bifurcation_solve as solve
import cube_incidence_symplectic_basis as cube_basis
import cube_incidence_symplectic_completion_gate as cube_gate
import density_oriented_pair_connection as density_connection
import personal_touch_activated_bond_basis as activated
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
CUBE_COEFFICIENTS = torch.tensor(
    cube_gate.COEFFICIENTS, dtype=torch.float64)
DENSITY_COEFFICIENTS = torch.tensor((1.0, 0.0), dtype=torch.float64)


def extra_potential(raw: torch.Tensor) -> torch.Tensor:
    return (sum(activated.energies(raw))
            + area_phase.raw_potential(raw)/native_time.CHI0**14)


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    potential = native_time.DT*(
        extra_potential(left)+extra_potential(right))/2
    return (
        native_box.interval(left, right, SIDE)-potential
        + native_time.B*native_time.CHI0*density_connection.interval(
            left, right, DENSITY_COEFFICIENTS)
        + cube_basis.interval(left, right, CUBE_COEFFICIENTS)
    )


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor) -> torch.Tensor:
    action = interval_action(previous, current)+interval_action(
        current, following)
    return torch.autograd.grad(action, current, create_graph=True)[0]


def expected_blocks():
    bare, mass0, gyro0, stiffness0 = solve.base_matrices()
    connection = solve.load_blocks("rho_Psi")
    cube = np.load(cube_gate.CACHE)
    natural = native_time.B*native_time.CHI0
    return bare, {
        "mass": mass0+natural*connection["mass"]+cube["mass"],
        "gyro": gyro0+natural*connection["gyro"]+cube["gyro"],
        "stiffness": (stiffness0+natural*connection["stiffness"]
                      + cube["stiffness"]),
    }


def main() -> None:
    bare, expected = expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    arrays = (rotation.T@bare["rest"], bare["rest"],
              rotation@bare["rest"])
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True) for value in arrays)
    rest = del_residual(*layers).detach().numpy()

    rng = np.random.default_rng(4006001)
    rows = []
    for sample in range(4):
        vector = rng.normal(size=bare["rest"].size)
        vector /= np.linalg.norm(vector)
        directions = (rotation.T@vector, vector, rotation@vector)
        images = []
        for selected in range(3):
            def function(value, argument=selected):
                trial = list(layers)
                trial[argument] = value
                return del_residual(*trial)
            _value, image = torch.autograd.functional.jvp(
                function, layers[selected],
                torch.tensor(directions[selected], dtype=torch.float64),
                create_graph=False, strict=False)
            images.append(image.detach().numpy())
        minus, zero, plus = images
        derived = {
            "mass": -0.5*native_time.DT*(plus+minus),
            "gyro": -(plus-minus),
            "stiffness": -(plus+zero+minus)/native_time.DT,
        }
        rows.append({
            "sample": sample,
            **{
                f"{kind}_relative_error": float(
                    np.linalg.norm(derived[kind]-expected[kind]@vector)
                    / max(np.linalg.norm(expected[kind]@vector), 1e-30))
                for kind in derived
            },
        })
    maximum = max(value for row in rows for key, value in row.items()
                  if key.endswith("relative_error"))
    result = {
        "status": "V4_CUBE_INCIDENCE_NONLINEAR_ACTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_terms": [
            "canonical V40 kinetic and potential action",
            "four native density-activated positive bond energies at unit weight",
            "area-bilinear phase square at 1/chi0^2",
            "common density-oriented pair connection at B*chi0",
            "unit cube-incidence symplectic completion",
        ],
        "rest_DEL_norm": float(np.linalg.norm(rest)),
        "directional_matrix_checks": rows,
        "maximum_directional_relative_error": maximum,
        "verdict": (
            "NONLINEAR_ACTION_REPRODUCES_LINEAR_CANDIDATE"
            if np.linalg.norm(rest) < 1e-8 and maximum < 1e-8
            else "NONLINEAR_ACTION_MISMATCH"),
        "claim_boundary": (
            "Action reconstruction and tangent identity only. It does not "
            "admit the research action or prove nonlinear stability, motion, "
            "current matching, or V4-006."),
    }
    (HERE/"cube_incidence_nonlinear_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
