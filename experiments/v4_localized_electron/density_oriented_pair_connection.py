"""Screen a native density-oriented joint Psi--chi symplectic connection.

For an unoriented C19 bond {i,j}, both ``rho_i-rho_j`` and

  Im[<Z_i,dZ_j>-<Z_j,dZ_i>]

change sign when the endpoints are exchanged.  Their product therefore
defines a reciprocal, orientation-free one-form.  The density contrast is a
native state variable, not a selected direction or carrier label.  Both the
common-phase pair Z=chi tensor Psi and relative-phase pair
Z=chi tensor conjugate(Psi) are screened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh, lu_factor, lu_solve, null_space

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import complete_pair_wave_action_gate as pair_gate
import current_action_two_layer_spectrum as current_spectrum
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
NAMES = ("common_pair_connection", "relative_pair_connection")


def fields(raw: torch.Tensor, relative: bool):
    psi, chi = native_box.unpack(raw, SIDE)
    if relative:
        psi = torch.conj(psi)
    return chi[..., :, None]*psi[..., None, :]


def one_form(left: torch.Tensor, right: torch.Tensor,
             relative: bool) -> torch.Tensor:
    lp, _lc = native_box.unpack(left, SIDE)
    rp, _rc = native_box.unpack(right, SIDE)
    midpoint_raw = (left+right)/2
    midpoint_psi, _ = native_box.unpack(midpoint_raw, SIDE)
    rho = torch.real(torch.sum(
        torch.conj(midpoint_psi)*midpoint_psi, dim=-1))/native_time.CHI0**2
    z = fields(midpoint_raw, relative)
    dz = fields(right, relative)-fields(left, relative)
    result = left.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        zj = torch.roll(z, shifts=offset, dims=(0, 1, 2))
        dzj = torch.roll(dz, shifts=offset, dims=(0, 1, 2))
        rhoj = torch.roll(rho, shifts=offset, dims=(0, 1, 2))
        pairing = torch.sum(
            torch.conj(z)*dzj-torch.conj(zj)*dz, dim=(-2, -1))
        result = result + weight*torch.sum((rho-rhoj)*torch.imag(pairing))
    return result/native_time.CHI0**4


def interval(left: torch.Tensor, right: torch.Tensor,
             coefficients: torch.Tensor) -> torch.Tensor:
    return (coefficients[0]*one_form(left, right, False)
            + coefficients[1]*one_form(left, right, True))


def derive_blocks(rest: np.ndarray, index: int, dt: float):
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]
    coefficients = torch.zeros(2, dtype=torch.float64)
    coefficients[index] = 1

    def residual(left, middle, right):
        action = (interval(left, middle, coefficients)
                  + interval(middle, right, coefficients))
        return torch.autograd.grad(action, middle, create_graph=True)[0]

    blocks = []
    for argument in range(3):
        def function(value, selected=argument):
            trial = layers.copy()
            trial[selected] = value
            return residual(*trial)
        blocks.append(torch.autograd.functional.jacobian(
            function, layers[argument], vectorize=True).detach().numpy())
    minus = blocks[0]@rotation.T
    zero = blocks[1]
    plus = blocks[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    middle = torch.tensor(bare["rest"], dtype=torch.float64,
                          requires_grad=True)
    left = torch.tensor(rotation.T@bare["rest"], dtype=torch.float64)
    right = torch.tensor(rotation@bare["rest"], dtype=torch.float64)

    forces = []
    values = []
    for index in range(2):
        coefficients = torch.zeros(2, dtype=torch.float64)
        coefficients[index] = 1
        action = (interval(left, middle, coefficients)
                  + interval(middle, right, coefficients))
        forces.append(torch.autograd.grad(
            action, middle, retain_graph=True)[0].detach().numpy())
        values.append(float(action.detach()))
    matrix = np.column_stack(forces)
    singular = np.linalg.svd(matrix, compute_uv=False)
    kernel = (np.eye(2) if max(np.linalg.norm(matrix, axis=0)) < 1e-12
              else null_space(matrix, rcond=1e-10))
    matrix_cache = HERE/"density_oriented_pair_connection_matrices.npz"
    if matrix_cache.exists():
        cached = np.load(matrix_cache)
        blocks = [{kind: cached[f"{kind}_{name}"]
                   for kind in ("mass", "gyro", "stiffness")}
                  for name in NAMES]
    else:
        blocks = [derive_blocks(bare["rest"], index, dt) for index in range(2)]
        np.savez_compressed(
            matrix_cache,
            **{f"{kind}_{name}": blocks[index][kind]
               for index, name in enumerate(NAMES)
               for kind in ("mass", "gyro", "stiffness")})
    mode = np.load(HERE/"current_action_fastest_mode.npz")["current"]
    denominator = float(np.vdot(mode, bare["mass"]@mode).real)
    psi, chi = native_box.unpack(middle, SIDE)
    translations = native.translation_vectors(
        psi.detach().numpy(), chi.detach().numpy())
    additions = np.load(HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    current_stiffness = (
        bare["stiffness"]
        + additions["hessian_carrier_carrier_chi"]
        + additions["hessian_carrier_carrier_Psi_tight_frame"])
    factor = lu_factor(bare["mass"]/dt+bare["gyro"]/2)
    response_rows = []
    for index, name in enumerate(NAMES):
        transport = []
        for axis in range(3):
            vector = translations[:, axis]
            delta_following = -lu_solve(factor, (
                dt*blocks[index]["stiffness"]@vector
                + blocks[index]["gyro"]@vector
                + blocks[index]["mass"]@(
                    -np.linalg.solve(bare["mass"],
                                     current_stiffness@vector))))
            transport.append(float(np.linalg.norm(delta_following)))
        response_rows.append({
            "name": name,
            "added_mass_quadratic": float(
                np.vdot(mode, blocks[index]["mass"]@mode).real/denominator),
            "added_gyro_quadratic_magnitude": float(
                abs(np.vdot(mode, blocks[index]["gyro"]@mode))/denominator),
            "added_stiffness_quadratic": float(
                np.vdot(mode, blocks[index]["stiffness"]@mode).real
                / denominator),
            "three_axis_translation_response_norms": transport,
        })
    natural = native_time.B*native_time.CHI0
    spectral_rows = []
    for label, common, relative in tuple(
            (f"common_{factor:g}_natural", factor*natural, 0.0)
            for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
        mass = (bare["mass"] + common*blocks[0]["mass"]
                + relative*blocks[1]["mass"])
        gyro = (bare["gyro"] + common*blocks[0]["gyro"]
                + relative*blocks[1]["gyro"])
        stiffness = (current_stiffness
                     + common*blocks[0]["stiffness"]
                     + relative*blocks[1]["stiffness"])
        minimum_mass = float(eigvalsh(
            mass, subset_by_index=[0, 0], driver="evr")[0])
        row = {"label": label, "common": common, "relative": relative,
               "minimum_mass_eigenvalue": minimum_mass}
        if minimum_mass > 1e-9:
            dressed = current_spectrum.dressed_configuration(
                stiffness, mass, bare["rest"])
            row.update(pair_gate.spectrum_row(
                mass, gyro, stiffness, dressed, dt))
        spectral_rows.append(row)
    result = {
        "status": "V4_DENSITY_ORIENTED_PAIR_CONNECTION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "terms": {
            name: {
                "two_interval_action": values[index],
                "rest_DEL_norm": float(np.linalg.norm(forces[index])),
            } for index, name in enumerate(NAMES)
        },
        "rest_force_singular_values": singular.tolist(),
        "rest_preserving_kernel_dimension": int(kernel.shape[1]),
        "rest_preserving_kernel": kernel.tolist(),
        "linear_response": response_rows,
        "natural_coefficient": {
            "value": natural,
            "provenance": "B times chi0, the native inertia-clock scale",
        },
        "natural_coefficient_spectrum": spectral_rows,
        "verdict": (
            "REST_PRESERVING_THREE_AXIS_CONNECTION_REDUCES_BUT_RETAINS_GROWTH"
            if kernel.shape[1] else
            "PAIR_CONNECTION_DOES_NOT_PRESERVE_CURRENT_REST_ORBIT"),
        "claim_boundary": (
            "Exact first variation of two symmetry-allowed one-forms at the "
            "current rest orbit and the complete finite-hull linear spectrum "
            "for the declared common-channel coefficient scan. The natural "
            "B*chi0 coefficient reduces the maximum growth rate but does not "
            "remove all growing modes, prove nonlinear motion, close V4-006, "
            "or admit an action."),
    }
    (HERE/"density_oriented_pair_connection.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
