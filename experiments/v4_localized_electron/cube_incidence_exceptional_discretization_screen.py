"""Screen exceptional-discretization directions against native pinning.

Exceptional discrete soliton models replace part of an onsite nonlinear
potential by a continuum-equivalent neighbor coupling.  This can remove the
Peierls--Nabarro barrier without introducing a preferred site or direction.

Here the local V40 density invariants are redistributed over the C19 bonds.
Every candidate is real, reciprocal, cubic, phase invariant, and uses only
native complex Psi and chi.  The calculation first finds combinations whose
first variation vanishes on the exact rest carrier, then asks whether their
Hessian images span the three translation-pinning forces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_box_transfer as box
import cube_incidence_nonlinear_action as action
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5
WEIGHT_SUM = 24.0


def rho(field: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(field) ** 2, dim=-1)


def bond_average(values, orbit: str) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    denominator = 0.0
    for offset, weight in native.DIRECTIONS:
        is_face = sum(abs(value) for value in offset) == 1
        if (orbit == "face") != is_face:
            continue
        total = total + weight * values(offset)
        denominator += weight
    return total / denominator


def shift(field: torch.Tensor, offset) -> torch.Tensor:
    return torch.roll(field, shifts=tuple(-value for value in offset),
                      dims=(0, 1, 2))


def inner_power(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    value = torch.sum(torch.conj(left) * right, dim=-1)
    return torch.abs(value) ** 2


def bilinear_power(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    value = torch.sum(left * right, dim=-1)
    return torch.abs(value) ** 2


def real_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    value = torch.sum(torch.conj(left) * right, dim=-1)
    return torch.real(value)


def area_power(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(torch.linalg.cross(left, right)) ** 2, dim=-1)


def candidate_values(raw: torch.Tensor) -> list[torch.Tensor]:
    psi, chi = box.unpack(raw, SIDE)
    rp = rho(psi)
    rc = rho(chi)
    defect = rc - 19.0**2

    onsite_psi = torch.sum(rp**2) / 4
    onsite_chi = torch.sum(defect**2) / 4
    onsite_cross = torch.sum(rc * rp) / 2
    onsite_parallel = torch.sum(inner_power(chi, psi))
    onsite_area = torch.sum(area_power(chi, psi))
    onsite_psi_bilinear = torch.sum(bilinear_power(psi, psi))
    onsite_chi_bilinear = torch.sum(bilinear_power(chi, chi))
    onsite_cross_bilinear = torch.sum(bilinear_power(chi, psi))
    onsite_psi_density_hop = torch.sum(rp**2)
    onsite_chi_density_hop = torch.sum(rc**2)
    onsite_chi_on_psi_hop = torch.sum(rc * rp)
    onsite_psi_on_chi_hop = torch.sum(rp * rc)
    onsite_scalar_overlap_hop = torch.sum(inner_power(chi, psi))
    sigma_psi = torch.sum(psi * psi, dim=-1)
    onsite_sigma_hop = torch.sum(torch.abs(sigma_psi)**2)

    result = []
    for orbit in ("face", "edge"):
        bond_psi = bond_average(lambda offset: torch.sum(
            rp * shift(rp, offset)), orbit) / 4
        bond_chi = bond_average(lambda offset: torch.sum(
            defect * shift(defect, offset)), orbit) / 4
        bond_cross = bond_average(lambda offset: torch.sum(
            rc * shift(rp, offset) + rp * shift(rc, offset)), orbit) / 4
        bond_parallel = bond_average(lambda offset: torch.sum(
            (inner_power(chi, shift(psi, offset))
             + inner_power(psi, shift(chi, offset))) / 2), orbit)
        bond_area = bond_average(lambda offset: torch.sum(
            (area_power(chi, shift(psi, offset))
             + area_power(psi, shift(chi, offset))) / 2), orbit)
        # Pure orientation terms have no onsite counterpart: a field is
        # parallel to itself.  They complete density products into
        # parallel+area channels.
        psi_orientation = bond_average(lambda offset: torch.sum(
            rp * shift(rp, offset)
            - inner_power(psi, shift(psi, offset))), orbit)
        chi_orientation = bond_average(lambda offset: torch.sum(
            rc * shift(rc, offset)
            - inner_power(chi, shift(chi, offset))), orbit)
        psi_bilinear = bond_average(lambda offset: torch.sum(
            bilinear_power(psi, shift(psi, offset))), orbit)
        chi_bilinear = bond_average(lambda offset: torch.sum(
            bilinear_power(chi, shift(chi, offset))), orbit)
        cross_bilinear = bond_average(lambda offset: torch.sum(
            (bilinear_power(chi, shift(psi, offset))
             + bilinear_power(psi, shift(chi, offset))) / 2), orbit)
        # Ablowitz--Ladik/exceptional-discretization channels omitted by the
        # preceding phase-insensitive basis: the local amplitude modulates a
        # coherent Hermitian neighbor transfer.  Subtracting the coincident
        # value makes each direction a redistribution of an existing onsite
        # invariant rather than a new preferred field configuration.
        psi_density_hop = bond_average(lambda offset: torch.sum(
            (rp + shift(rp, offset)) / 2
            * real_inner(psi, shift(psi, offset))), orbit)
        chi_density_hop = bond_average(lambda offset: torch.sum(
            (rc + shift(rc, offset)) / 2
            * real_inner(chi, shift(chi, offset))), orbit)
        chi_on_psi_hop = bond_average(lambda offset: torch.sum(
            (rc + shift(rc, offset)) / 2
            * real_inner(psi, shift(psi, offset))), orbit)
        psi_on_chi_hop = bond_average(lambda offset: torch.sum(
            (rp + shift(rp, offset)) / 2
            * real_inner(chi, shift(chi, offset))), orbit)
        scalar_overlap_hop = bond_average(lambda offset: torch.sum(
            torch.real(
                torch.conj(torch.sum(torch.conj(chi) * psi, dim=-1))
                * shift(torch.sum(torch.conj(chi) * psi, dim=-1), offset))),
            orbit)
        sigma_hop = bond_average(lambda offset: torch.sum(
            torch.real(torch.conj(sigma_psi) * shift(sigma_psi, offset))),
            orbit)
        result.extend((
            bond_psi - onsite_psi,
            bond_chi - onsite_chi,
            bond_cross - onsite_cross,
            bond_parallel - onsite_parallel,
            bond_area - onsite_area,
            psi_orientation,
            chi_orientation,
            psi_bilinear - onsite_psi_bilinear,
            chi_bilinear - onsite_chi_bilinear,
            cross_bilinear - onsite_cross_bilinear,
            psi_density_hop - onsite_psi_density_hop,
            chi_density_hop - onsite_chi_density_hop,
            chi_on_psi_hop - onsite_chi_on_psi_hop,
            psi_on_chi_hop - onsite_psi_on_chi_hop,
            scalar_overlap_hop - onsite_scalar_overlap_hop,
            sigma_hop - onsite_sigma_hop,
        ))
    return result


NAMES = [
    f"{orbit}_{name}"
    for orbit in ("face", "edge")
    for name in (
        "Psi_quartic_redistribution",
        "chi_vacuum_redistribution",
        "Psi_chi_density_redistribution",
        "parallel_frame_redistribution",
        "area_frame_redistribution",
        "Psi_orientation_bond",
        "chi_orientation_bond",
        "Psi_bilinear_redistribution",
        "chi_bilinear_redistribution",
        "Psi_chi_bilinear_redistribution",
        "Psi_density_coherent_hopping",
        "chi_density_coherent_hopping",
        "chi_density_Psi_coherent_hopping",
        "Psi_density_chi_coherent_hopping",
        "scalar_overlap_coherent_hopping",
        "Psi_sigma_coherent_hopping",
    )
]


def residual_summary(vector: np.ndarray, carrier: np.ndarray) -> dict:
    rows = []
    for axis in range(3):
        part = vector[axis * carrier.size:(axis + 1) * carrier.size]
        dpsi, dchi = native.unpack(part, SIDE)
        psi_power = np.sum(abs(dpsi) ** 2)
        chi_power = np.sum(abs(dchi) ** 2)
        total = max(float(psi_power + chi_power), 1e-300)
        component_power = (np.sum(abs(dpsi) ** 2, axis=(0, 1, 2))
                           + np.sum(abs(dchi) ** 2, axis=(0, 1, 2)))
        carrier_mask = native.norm2(native.unpack(carrier, SIDE)[0]) > 1e-12
        site_power = native.norm2(dpsi) + native.norm2(dchi)
        rows.append({
            "axis": axis,
            "Psi_power_fraction": float(psi_power / total),
            "chi_power_fraction": float(chi_power / total),
            "C3_component_power_fractions": [
                float(value / total) for value in component_power],
            "carrier_site_power_fraction": float(
                np.sum(site_power[carrier_mask]) / total),
        })
    return {"axis_rows": rows}


def main() -> None:
    bare, matrices = action.expected_blocks()
    rest = torch.tensor(bare["rest"], dtype=torch.float64,
                        requires_grad=True)
    psi, chi = box.unpack(rest, SIDE)
    translations = native.translation_vectors(
        psi.detach().numpy(), chi.detach().numpy())

    gradients = []
    responses = []
    values = candidate_values(rest)
    for value in values:
        gradient = torch.autograd.grad(value, rest, create_graph=True)[0]
        gradients.append(gradient.detach().numpy())
        axis_images = []
        for axis in range(3):
            direction = torch.tensor(translations[:, axis],
                                     dtype=torch.float64)
            image = torch.autograd.grad(
                torch.dot(gradient, direction), rest,
                retain_graph=True)[0]
            axis_images.append(image.detach().numpy())
        responses.append(np.concatenate(axis_images))

    force = np.column_stack(gradients)
    u, singular_force, vh = np.linalg.svd(force, full_matrices=False)
    threshold = max(singular_force[0], 1.0) * 1e-10
    force_rank = int(np.sum(singular_force > threshold))
    rest_null = vh[force_rank:].T

    stiffness = matrices["stiffness"]
    target = np.concatenate([
        -(stiffness @ translations[:, axis]) for axis in range(3)])
    response = np.column_stack(responses)
    if rest_null.shape[1]:
        admissible = response @ rest_null
        null_coefficients, *_ = np.linalg.lstsq(
            admissible, target, rcond=1e-10)
        coefficients = rest_null @ null_coefficients
        residual = target - response @ coefficients
        admissible_singular = np.linalg.svd(admissible, compute_uv=False)
        admissible_rank = int(np.sum(
            admissible_singular
            > max(admissible_singular[0], 1.0) * 1e-10))
    else:
        coefficients = np.zeros(len(NAMES))
        residual = target.copy()
        admissible_rank = 0

    # Also report the unconstrained oracle to distinguish insufficient
    # response from incompatibility with the exact rest branch.
    oracle_coefficients, *_ = np.linalg.lstsq(response, target, rcond=1e-10)
    oracle_residual = target - response @ oracle_coefficients
    result = {
        "status": "V4_EXCEPTIONAL_C19_DISCRETIZATION_SCREEN",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "candidate_actions": NAMES,
        "rest_force_rank": force_rank,
        "rest_force_nullity": int(rest_null.shape[1]),
        "rest_force_singular_values": [float(value)
                                       for value in singular_force],
        "rest_preserving_translation_response_rank": admissible_rank,
        "unrepaired_translation_target_norm": float(np.linalg.norm(target)),
        "best_rest_preserving_relative_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-300)),
        "best_unconstrained_relative_residual": float(
            np.linalg.norm(oracle_residual)
            / max(np.linalg.norm(target), 1e-300)),
        "unrepaired_target_channels": residual_summary(
            target, bare["rest"]),
        "rest_preserving_residual_channels": residual_summary(
            residual, bare["rest"]),
        "diagnostic_rest_preserving_coefficients": {
            name: float(value) for name, value in zip(
                NAMES, coefficients, strict=True)
        },
        "diagnostic_unconstrained_coefficients": {
            name: float(value) for name, value in zip(
                NAMES, oracle_coefficients, strict=True)
        },
        "coefficients_admitted": False,
        "verdict": (
            "EXCEPTIONAL_C19_BASIS_HAS_REST_PRESERVING_CLOSING_DIRECTION"
            if np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-300)
            < 0.05 else
            "EXCEPTIONAL_C19_BASIS_DOES_NOT_CLOSE_PINNING"),
        "claim_boundary": (
            "First-order rest preservation and translation-tangent response "
            "only. No coefficient is admitted; a closing direction would "
            "still require bounded energy, a re-solved nonlinear branch, "
            "Dirac current, and long movement tests."),
    }
    (HERE / "cube_incidence_exceptional_discretization_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
