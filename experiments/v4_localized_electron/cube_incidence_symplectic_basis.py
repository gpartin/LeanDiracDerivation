"""Derive the carrier-activated C19 incidence symplectic basis.

For each native joint Psi--chi composite F, this screen compares

  sum_i a_i^2 Im <F_i, dF_i>

with the C19-gradient Berry form

  1/2 sum_(i,n) w_n a_i a_(i+n)
      Im <F_(i+n)-F_i, dF_(i+n)-dF_i>,

where a_i=rho_Psi,i/chi0^2.  The activation is a native scalar and makes the
internal cube graph emerge without naming a carrier, corner, axis, copy or
Dirac matrix.  Rest-preserving combinations are obtained only from the exact
DEL force nullspace.  Their linear response is then tested on the independently
identified common Psi--chi 001 residual.

This is an algebraic response screen, not an admitted action.
"""

from __future__ import annotations

import fractions
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_geometry_bifurcation_residual as residual_mode
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
REPRESENTATIONS = (
    "tensor_common",
    "tensor_relative",
    "area_common",
    "area_relative",
)
BASIS = tuple(
    f"{representation}_{geometry}"
    for representation in REPRESENTATIONS
    for geometry in ("local", "C19_gradient")
)


def representation(raw: torch.Tensor, name: str):
    psi, chi = native_box.unpack(raw, SIDE)
    if name.endswith("relative"):
        psi = torch.conj(psi)
    if name.startswith("tensor"):
        return chi[..., :, None]*psi[..., None, :]/native_time.CHI0
    return torch.linalg.cross(chi, psi, dim=-1)/native_time.CHI0


def activation(raw: torch.Tensor):
    psi, _chi = native_box.unpack(raw, SIDE)
    return torch.real(torch.sum(torch.conj(psi)*psi, dim=-1))/(
        native_time.CHI0**2)


def one_form(left: torch.Tensor, right: torch.Tensor,
             representation_name: str, geometry: str):
    midpoint = (left+right)/2
    field = representation(midpoint, representation_name)
    dfield = representation(right, representation_name)-representation(
        left, representation_name)
    active = activation(midpoint)
    if geometry == "local":
        pairing = torch.sum(torch.conj(field)*dfield,
                            dim=tuple(range(3, field.ndim)))
        return torch.sum(active**2*torch.imag(pairing))
    result = left.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        neighbor = torch.roll(field, shifts=offset, dims=(0, 1, 2))
        dneighbor = torch.roll(dfield, shifts=offset, dims=(0, 1, 2))
        neighbor_active = torch.roll(
            active, shifts=offset, dims=(0, 1, 2))
        difference = neighbor-field
        ddifference = dneighbor-dfield
        pairing = torch.sum(
            torch.conj(difference)*ddifference,
            dim=tuple(range(3, difference.ndim)))
        result = result+0.5*weight*torch.sum(
            active*neighbor_active*torch.imag(pairing))
    return result


def interval(left: torch.Tensor, right: torch.Tensor,
             coefficients: torch.Tensor):
    terms = []
    for representation_name in REPRESENTATIONS:
        for geometry in ("local", "C19_gradient"):
            terms.append(one_form(
                left, right, representation_name, geometry))
    return sum(coefficients[index]*term
               for index, term in enumerate(terms))


def del_residual(left: torch.Tensor, middle: torch.Tensor,
                 right: torch.Tensor, coefficients: torch.Tensor):
    action = interval(left, middle, coefficients)+interval(
        middle, right, coefficients)
    return torch.autograd.grad(action, middle, create_graph=True)[0]


def force_column(layers, index: int):
    coefficients = torch.zeros(len(BASIS), dtype=torch.float64)
    coefficients[index] = 1
    return del_residual(*layers, coefficients).detach().numpy()


def directional_blocks(layers, rotation: np.ndarray, vector: np.ndarray,
                       index: int):
    coefficients = torch.zeros(len(BASIS), dtype=torch.float64)
    coefficients[index] = 1

    def function(left, middle, right):
        return del_residual(left, middle, right, coefficients)

    directions = (
        torch.tensor(rotation.T@vector, dtype=torch.float64),
        torch.tensor(vector, dtype=torch.float64),
        torch.tensor(rotation@vector, dtype=torch.float64),
    )
    images = []
    for selected in range(3):
        def selected_function(value, argument=selected):
            trial = list(layers)
            trial[argument] = value
            return function(*trial)
        _value, image = torch.autograd.functional.jvp(
            selected_function, layers[selected], directions[selected],
            create_graph=False, strict=False)
        images.append(image.detach().numpy())
    minus, zero, plus = images
    dt = native_time.DT
    return {
        "mass_on_mode": -0.5*dt*(plus+minus),
        "gyro_on_mode": -(plus-minus),
        "stiffness_on_mode": -(plus+zero+minus)/dt,
    }


def main() -> None:
    bare = np.load(canonical.CACHE)
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64,
                                requires_grad=True)
                   for value in (rotation.T@bare["rest"], bare["rest"],
                                 rotation@bare["rest"]))
    forces = np.column_stack([
        force_column(layers, index) for index in range(len(BASIS))])
    force_norms = np.linalg.norm(forces, axis=0)
    normalized = forces/np.maximum(force_norms, 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    kernel = null_space(normalized, rcond=1e-10)

    fit = json.loads((HERE/"cube_geometry_bifurcation_solve.json").read_text())
    coefficients = np.asarray(
        fit["continuous_solution"]["coefficients_in_B_chi0_units"])
    rest, _blocks, matrices = residual_mode.assemble(coefficients)
    _multiplier, complex_target = residual_mode.fastest_mode(rest, matrices)
    pivot = int(np.argmax(abs(complex_target)))
    phase = float(np.angle(complex_target[pivot]))
    target = np.real(np.exp(-1j*phase)*complex_target)
    target /= max(np.linalg.norm(target), 1e-30)
    responses = [directional_blocks(layers, rotation, target, index)
                 for index in range(len(BASIS))]

    response_rows = []
    for index, name in enumerate(BASIS):
        row = {"name": name, "rest_DEL_norm": float(force_norms[index])}
        for kind, image in responses[index].items():
            row[f"{kind}_norm"] = float(np.linalg.norm(image))
            row[f"{kind}_target_overlap"] = float(
                abs(np.vdot(target, image))/max(
                    np.linalg.norm(target)*np.linalg.norm(image), 1e-30))
        response_rows.append(row)

    kernel_rows = []
    combined_images = {}
    for kind in ("mass_on_mode", "gyro_on_mode", "stiffness_on_mode"):
        combined_images[kind] = np.column_stack(
            [row[kind] for row in responses])@kernel
    for column in range(kernel.shape[1]):
        vector = kernel[:, column]
        # Undo column normalization to express the null vector in the original
        # action basis; normalize only for reporting.
        action_coefficients = vector/np.maximum(force_norms, 1e-30)
        action_coefficients /= max(np.max(abs(action_coefficients)), 1e-30)
        kernel_rows.append({
            "coefficients": {
                name: float(value) for name, value in zip(
                    BASIS, action_coefficients, strict=True)},
            "direct_rest_DEL_norm": float(np.linalg.norm(
                forces@action_coefficients)),
            "mass_response_norm": float(np.linalg.norm(
                sum(action_coefficients[index]
                    * responses[index]["mass_on_mode"]
                    for index in range(len(BASIS))))),
            "gyro_response_norm": float(np.linalg.norm(
                sum(action_coefficients[index]
                    * responses[index]["gyro_on_mode"]
                    for index in range(len(BASIS))))),
            "stiffness_response_norm": float(np.linalg.norm(
                sum(action_coefficients[index]
                    * responses[index]["stiffness_on_mode"]
                    for index in range(len(BASIS))))),
        })

    completion_rows = []
    for selected in (1, 3, 5, 7):
        others = [index for index in range(len(BASIS)) if index != selected]
        solved, *_ = np.linalg.lstsq(
            forces[:, others], -forces[:, selected], rcond=1e-12)
        action_coefficients = np.zeros(len(BASIS))
        action_coefficients[selected] = 1.0
        action_coefficients[others] = solved
        images = {
            kind: sum(action_coefficients[index]*responses[index][kind]
                      for index in range(len(BASIS)))
            for kind in ("mass_on_mode", "gyro_on_mode",
                         "stiffness_on_mode")
        }
        completion_rows.append({
            "selected_gradient": BASIS[selected],
            "coefficients": {
                name: float(value) for name, value in zip(
                    BASIS, action_coefficients, strict=True)},
            "rational_coefficients_denominator_at_most_48": {
                name: str(fractions.Fraction(float(value)).limit_denominator(48))
                for name, value in zip(BASIS, action_coefficients, strict=True)},
            "rest_DEL_norm": float(np.linalg.norm(
                forces@action_coefficients)),
            "rest_DEL_relative_to_selected": float(np.linalg.norm(
                forces@action_coefficients)/max(force_norms[selected], 1e-30)),
            "mass_response_norm": float(np.linalg.norm(
                images["mass_on_mode"])),
            "gyro_response_norm": float(np.linalg.norm(
                images["gyro_on_mode"])),
            "stiffness_response_norm": float(np.linalg.norm(
                images["stiffness_on_mode"])),
        })

    result = {
        "status": "V4_CUBE_INCIDENCE_SYMPLECTIC_BASIS",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": list(BASIS),
        "action_definition": (
            "carrier-density-activated local and C19-gradient Berry forms "
            "for native common/relative tensor and area composites"),
        "cube_geometry": {
            "C19_face_weight": 2,
            "C19_edge_weight": 1,
            "target_character": "001",
            "target_restricted_C19_eigenvalue": 8,
        },
        "force_rank": int(np.linalg.matrix_rank(normalized, tol=1e-10)),
        "normalized_force_singular_values": singular.tolist(),
        "rest_preserving_kernel_dimension": int(kernel.shape[1]),
        "basis_rows": response_rows,
        "rest_preserving_kernel_rows": kernel_rows,
        "single_gradient_minimum_local_completions": completion_rows,
        "verdict": (
            "CUBE_INCIDENCE_SYMPLECTIC_RESPONSE_SURVIVES"
            if any(row["gyro_response_norm"] > 1e-8
                   and row["direct_rest_DEL_norm"] < 1e-8
                   for row in kernel_rows)
            else "CUBE_INCIDENCE_SYMPLECTIC_BASIS_NO_RESPONSE"),
        "claim_boundary": (
            "Rest-force nullspace and one-mode differential response only. "
            "It does not admit an action, establish coefficient provenance, "
            "or close the full spectrum, nonlinear motion, or V4-006."),
    }
    (HERE/"cube_incidence_symplectic_basis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
