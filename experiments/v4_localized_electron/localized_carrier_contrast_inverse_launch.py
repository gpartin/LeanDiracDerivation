"""Solve the native previous layer for a one-edge carrier displacement.

The exact carrier-minus-vacuum profile and both cubical copies span physical
translation, but a rigidly shifted previous layer gives the wrong phase-current
sign in two directions.  This audit removes that guessed launch.  At the exact
rest orbit it linearizes the complete discrete Euler--Lagrange equation

    R(q_minus, q_zero, q_plus) = 0

and, for each signed axis, solves

    D_minus R [delta q_minus] = -D_plus R [delta q_plus]

where ``delta q_plus`` is the literal one-edge displacement of the following
carrier.  The previous perturbation is restricted only to translated copies
of the exact two-copy carrier contrast.  The Dirac equation, a target current,
and a runtime projection do not enter the solve.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vmap

import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank
import localized_carrier_contrast_full_current as current_audit
import localized_carrier_contrast_translation as contrast
import localized_embedding_native_closure as local
import localized_embedding_translation_tangent as tangent
import transfer_neighbor_reflection_rest_spectrum as reflection


HERE = Path(__file__).resolve().parent
SIDE = 5
DT = 1.0 / 256.0
ORIGIN = SIDE // 2 - 1


def centers(kind: str) -> list[tuple[int, int, int]]:
    if kind == "source_plus_faces":
        return [(0, 0, 0)] + [
            tuple(sign if axis == index else 0 for index in range(3))
            for axis in range(3) for sign in (-1, 1)
        ]
    if kind == "three_by_three_by_three":
        return list(itertools.product((-1, 0, 1), repeat=3))
    raise ValueError(kind)


def profile_columns(
        basis: np.ndarray, psi_profile: np.ndarray,
        chi_profile: np.ndarray,
        offset: tuple[int, int, int]) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
            site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                         for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[site] += amplitude * psi_profile[corner]
            dchi[site] += amplitude * chi_profile[corner]
        columns.extend((rank.pack(dpsi, dchi),
                        rank.pack(1j*dpsi, 1j*dchi)))
    return columns


def envelope(psi_phase: complex, center_kind: str) -> np.ndarray:
    psi_profile, chi_profile = contrast.carrier_contrast()
    flavors = local.flavor_bases()
    raw = np.column_stack([
        column
        for basis in (flavors["plus"], flavors["minus"])
        for center in centers(center_kind)
        for column in profile_columns(
            basis, psi_phase*psi_profile, chi_profile, center)
    ])
    return tangent.orth(raw)


def native_first_shell_envelope() -> np.ndarray:
    """All native Psi/chi coordinates on the carrier plus first shell.

    The 3x3x3 translated carrier centers occupy site coordinates 0 through 3
    in the side-five box.  This control determines whether the residual left
    by the two-copy envelope is an internal/radiative first-shell direction,
    rather than evidence for an action change.
    """
    shape = (SIDE, SIDE, SIDE, 4, 3)
    indices = np.arange(np.prod(shape)).reshape(shape)
    selected = indices[0:4, 0:4, 0:4].reshape(-1)
    return np.eye(np.prod(shape), dtype=float)[:, selected]


def rest_layers() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    amplitude_sq = rank.R**2 / (2 + 54/rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    return (rank.pack(np.exp(-1j*theta)*psi, chi),
            rank.pack(psi, chi),
            rank.pack(np.exp(1j*theta)*psi, chi), theta)


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return reflection.interval_action(left, right, reflection.COEFFICIENT)


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current) + interval_action(current,
                                                                 following)


del_residual = grad(three_layer_action, argnums=1)


def batched_jvp(argument: str, base: tuple[torch.Tensor, ...],
                vectors: np.ndarray) -> np.ndarray:
    previous, current, following = base
    vector_tensor = torch.tensor(vectors.T, dtype=torch.float64)
    if argument == "previous":
        function = lambda value: del_residual(value, current, following)
        point = previous
    elif argument == "following":
        function = lambda value: del_residual(previous, current, value)
        point = following
    else:
        raise ValueError(argument)

    def apply(vector: torch.Tensor) -> torch.Tensor:
        return jvp(function, (point,), (vector,))[1]

    images = vmap(apply, chunk_size=8)(vector_tensor)
    return images.detach().numpy().T


def shifted_difference(raw: np.ndarray, axis: int, sign: int) -> np.ndarray:
    psi, chi = rank.native.unpack(raw, SIDE)
    return rank.pack(np.roll(psi, sign, axis=axis)-psi,
                     np.roll(chi, sign, axis=axis)-chi)


def field_power(raw: np.ndarray) -> dict:
    psi, chi = rank.native.unpack(raw, SIDE)
    psi_power = float(np.sum(abs(psi)**2))
    chi_power = float(np.sum(abs(chi)**2))
    total = max(psi_power+chi_power, 1e-30)
    return {"Psi": psi_power/total, "chi": chi_power/total,
            "chi_vacuum_normal": float(np.sum(abs(chi[..., 2])**2))/total}


def residual_location(raw: np.ndarray) -> dict:
    psi, chi = rank.native.unpack(raw, SIDE)
    psi_site = np.sum(abs(psi)**2, axis=-1)
    chi_site = np.sum(abs(chi)**2, axis=-1)
    field = "Psi" if np.max(psi_site) >= np.max(chi_site) else "chi"
    site_power = psi_site if field == "Psi" else chi_site
    site = tuple(int(value) for value in np.unravel_index(
        np.argmax(site_power), site_power.shape))
    components = psi[site] if field == "Psi" else chi[site]
    component = int(np.argmax(abs(components)))
    return {"field": field, "site": list(site), "component": component,
            "absolute_amplitude": float(abs(components[component]))}


def flux_rows(previous: np.ndarray, launch: np.ndarray,
              axis: int, sign: int, step: float = 1e-5) -> list[dict]:
    _, current, _, _ = rest_layers()
    mask = current_audit.cut_mask(axis, sign)
    rows = []
    for mode in current_audit.MODES:
        baseline = current_audit.interval_flux(previous, current, mask, mode)
        minus = current_audit.interval_flux(
            previous-step*launch, current, mask, mode)
        plus = current_audit.interval_flux(
            previous+step*launch, current, mask, mode)
        derivative = (plus-minus)/(2*step)
        rows.append({
            "phase_mode": mode,
            "rest_cut_flux": baseline,
            "destination_inflow_response": float(derivative),
            "signed_axis_current_response": float(sign*derivative),
            "flows_into_receiver": bool(derivative > 0),
        })
    return rows


def receiver_charge_rows(current: np.ndarray, following: np.ndarray,
                         target: np.ndarray, axis: int, sign: int,
                         step: float = 1e-5) -> list[dict]:
    """Differentiate the actual current-to-following receiver interval.

    ``interval_flux`` is minus the regional Noether-charge change per unit
    time.  The old audit called it inflow; here the sign is reported exactly.
    The resting carrier has positive common and Psi phase charge, so positive
    regional charge change is the receiver-arrival criterion.
    """
    mask = current_audit.cut_mask(axis, sign)
    rows = []
    for mode in current_audit.MODES:
        baseline = current_audit.interval_flux(
            current, following, mask, mode)
        minus = current_audit.interval_flux(
            current, following-step*target, mask, mode)
        plus = current_audit.interval_flux(
            current, following+step*target, mask, mode)
        outward_rate = (plus-minus)/(2*step)
        charge_change = -DT*outward_rate
        rows.append({
            "phase_mode": mode,
            "rest_outward_rate": baseline,
            "outward_rate_response": float(outward_rate),
            "regional_charge_change_response": float(charge_change),
            "positive_charge_arrives": bool(charge_change > 0),
        })
    return rows


def main() -> None:
    rank.lock_inertia()
    rest.SIDE = SIDE
    rest.DT = DT
    current_audit.SIDE = SIDE
    current_audit.DT = DT
    previous, current, following, theta = rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    rows = []
    for center_kind in ("source_plus_faces", "three_by_three_by_three",
                        "full_native_first_shell"):
        q_previous = (native_first_shell_envelope()
                      if center_kind == "full_native_first_shell" else
                      envelope(np.exp(-1j*theta), center_kind))
        previous_images = batched_jvp("previous", base, q_previous)
        for axis in range(3):
            for sign in (-1, 1):
                target = shifted_difference(following, axis, sign)
                target_image = batched_jvp(
                    "following", base, target[:, None])[:, 0]
                coefficients, _, solve_rank, singular = np.linalg.lstsq(
                    previous_images, -target_image, rcond=1e-11)
                launch = q_previous @ coefficients
                equation_residual = previous_images @ coefficients + target_image
                relative = float(np.linalg.norm(equation_residual)
                                 / max(np.linalg.norm(target_image), 1e-30))
                rigid = shifted_difference(previous, axis, -sign)
                rigid_scale = float(np.dot(rigid, launch)
                                    / max(np.dot(rigid, rigid), 1e-30))
                rigid_relative = float(np.linalg.norm(
                    launch-rigid_scale*rigid)
                    / max(np.linalg.norm(launch), 1e-30))
                launch_currents = flux_rows(previous, launch, axis, sign)
                receiver = receiver_charge_rows(
                    current, following, target, axis, sign)
                common = next(row for row in receiver
                              if row["phase_mode"] == "common")
                rows.append({
                    "previous_center_family": center_kind,
                    "axis": axis,
                    "sign": sign,
                    "target_next_layer_kind": (
                        "literal one-edge carrier displacement"),
                    "previous_envelope_real_rank": int(q_previous.shape[1]),
                    "inverse_solve_rank": int(solve_rank),
                    "inverse_solve_condition": float(
                        singular[0]/max(singular[-1], 1e-300)),
                    "DEL_relative_residual": relative,
                    "DEL_residual_field_power_fractions": field_power(
                        equation_residual),
                    "DEL_first_maximum": residual_location(
                        equation_residual),
                    "launch_norm": float(np.linalg.norm(launch)),
                    "launch_field_power_fractions": field_power(launch),
                    "best_rigid_reverse_shift_scale": rigid_scale,
                    "distance_from_best_rigid_reverse_shift": rigid_relative,
                    "preceding_interval_phase_current_rows": launch_currents,
                    "receiver_interval_charge_rows": receiver,
                    "common_positive_charge_arrives": common[
                        "positive_charge_arrives"],
                })

    expanded = [row for row in rows if row["previous_center_family"]
                == "full_native_first_shell"]
    common_pass = all(row["common_positive_charge_arrives"]
                      for row in expanded)
    closure_pass = max(row["DEL_relative_residual"]
                       for row in expanded) < 1e-8
    result = {
        "status": "V4_CARRIER_CONTRAST_INVERSE_LAUNCH_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "complete transfer-softened interval action plus half-weight "
            "transfer-activated chi bond and 1/(2 chi0) frame reflection"),
        "method": (
            "solve the linearized native discrete Euler-Lagrange equation "
            "backward for the previous committed layer inside the exact "
            "two-copy carrier-minus-vacuum envelope"),
        "box": "side-five periodic response control",
        "dt": DT,
        "rows": rows,
        "all_six_DEL_inverse_solves_close": closure_pass,
        "expanded_shell_all_six_common_positive_charges_arrive": common_pass,
        "verdict": (
            "NATIVE_INVERSE_LAUNCH_CLOSES_EDGE_SNAP_AND_CURRENT"
            if closure_pass and common_pass else
            "NATIVE_INVERSE_LAUNCH_REMAINS_OPEN"),
        "claim_boundary": (
            "This is a linear inverse launch and current test. It does not "
            "establish a finite nonlinear hop, repeated translation, rest "
            "stability, or equality with the Dirac current."),
        "corrective_action": (
            "If the DEL inverse closes but current fails, derive the exact "
            "phase-density continuity identity before changing the action. "
            "If the inverse itself fails, decompose its residual by field "
            "and site to identify the missing native response."),
    }
    output = HERE / "localized_carrier_contrast_inverse_launch.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
