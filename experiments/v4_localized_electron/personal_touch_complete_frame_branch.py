"""Continue the exact carrier under the complete ordered-frame bond action.

The new energy is the Hilbert--Schmidt C19 difference of the full ordered
Psi/chi dyad tensor.  It is positive, reciprocal, cubic covariant, and
invariant under the independent global field phases.  All fields and the two
phase advances are re-solved; no target corner force is imposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eig

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_transfer_oscillator_gate as oscillator
import localized_stability as native
import personal_touch_face_identity_basis as face


HERE = Path(__file__).resolve().parent
SIDE = 3
STRENGTHS = (0.001, 0.01, 0.1, 1.0)
FRAME_ENERGY = None
FRAME_INTERVAL = None
FRAME_ENERGY_NAME = "complete_ordered_frame"
BASE_EVEN_STRENGTH = 0.0
BASE_DENSITY_STRENGTH = 0.0
BASE_FACE_SCALE = 0.0
BASE_FACE_TERMS = (
    (0.14705479982595762, "diagonal", "symmetric"),
    (0.14705479982595762, "diagonal", "antisymmetric"),
    (0.075, "diagonal_reversed", "full"),
    (0.075, "diagonal_conjugate", "full"),
    (0.10294520017404238, "edge_commutator", "full"),
)


def norm2(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value) * value, dim=(-2, -1)))


def dyad(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., :, None] * torch.conj(right[..., None, :])


def complete_frame_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    pieces = (dyad(psi, psi), dyad(chi, chi),
              np.sqrt(2.0) * dyad(psi, chi))
    energy = torch.zeros((), dtype=torch.float64)
    for piece in pieces:
        for offset, weight in native.DIRECTIONS:
            delta = piece - torch.roll(piece, shifts=offset,
                                       dims=(0, 1, 2))
            energy += weight * torch.sum(norm2(delta))
    return energy / (4 * small.CHI0**2)


def ordered_frame_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    pieces = (dyad(psi, psi), dyad(chi, chi))
    energy = torch.zeros((), dtype=torch.float64)
    for piece in pieces:
        for offset, weight in native.DIRECTIONS:
            delta = piece - torch.roll(piece, shifts=offset,
                                       dims=(0, 1, 2))
            energy += weight * torch.sum(norm2(delta))
    return energy / (4 * small.CHI0**2)


def chi_orientation_energy(raw: torch.Tensor) -> torch.Tensor:
    """C19 stiffness of the native chi orientation tensor.

    The tensor chi chi^dagger is invariant under a local phase of chi and
    retains its internal C3 direction.  Unlike W = chi wedge Psi, it remains
    observable in the Psi=0 exterior.  The bidirectional C19 normalization is
    identical to the ordered-frame action used elsewhere in this audit.
    """
    _psi, chi = box.unpack(raw, SIDE)
    piece = dyad(chi, chi)
    energy = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        delta = piece - torch.roll(piece, shifts=offset,
                                   dims=(0, 1, 2))
        energy += weight * torch.sum(norm2(delta))
    return energy / (4 * small.CHI0**2)


def plane_frame_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    plane = dyad(psi, psi) + dyad(chi, chi)
    energy = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        delta = plane - torch.roll(plane, shifts=offset, dims=(0, 1, 2))
        energy += weight * torch.sum(norm2(delta))
    return energy / (4 * small.CHI0**2)


def zero_energy(raw: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float64)


def plane_kinetic_interval(left: torch.Tensor,
                           right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_plane = dyad(left_psi, left_psi) + dyad(left_chi, left_chi)
    right_plane = dyad(right_psi, right_psi) + dyad(right_chi, right_chi)
    delta = right_plane-left_plane
    return torch.sum(norm2(delta))/(4*small.DT*small.CHI0**2)


def area_berry_interval(left: torch.Tensor,
                        right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_area = torch.linalg.cross(left_chi, left_psi, dim=-1)
    right_area = torch.linalg.cross(right_chi, right_psi, dim=-1)
    midpoint = (left_area+right_area)/2
    increment = right_area-left_area
    return torch.sum(torch.imag(torch.conj(midpoint)*increment)) / (
        small.CHI0**2)


def cross_berry_interval(left: torch.Tensor,
                         right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_cross = dyad(left_chi, left_psi)
    right_cross = dyad(right_chi, right_psi)
    midpoint = (left_cross+right_cross)/2
    increment = right_cross-left_cross
    return torch.sum(torch.imag(torch.conj(midpoint)*increment)) / (
        small.CHI0**2)


def cross_oscillator_interval(left: torch.Tensor,
                              right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_cross = dyad(left_chi, left_psi)
    right_cross = dyad(right_chi, right_psi)
    difference = right_cross-left_cross
    kinetic = torch.sum(norm2(difference))
    potential = torch.sum(norm2(left_cross)+norm2(right_cross))
    normalization = small.CHI0**4
    return (kinetic/(4*small.DT*normalization)
            - small.DT*small.CHI0**2*potential/(8*normalization))


def cross_bond_oscillator_interval(left: torch.Tensor,
                                   right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_cross = dyad(left_chi, left_psi)
    right_cross = dyad(right_chi, right_psi)
    kinetic = torch.zeros((), dtype=torch.float64)
    potential = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        left_bond = left_cross-torch.roll(
            left_cross, shifts=offset, dims=(0, 1, 2))
        right_bond = right_cross-torch.roll(
            right_cross, shifts=offset, dims=(0, 1, 2))
        kinetic += weight*torch.sum(norm2(right_bond-left_bond))
        potential += weight*torch.sum(norm2(left_bond)+norm2(right_bond))
    normalization = small.CHI0**4
    return (kinetic/(8*small.DT*normalization)
            - small.DT*small.CHI0**2*potential/(16*normalization))


def cross_bond_berry_interval(left: torch.Tensor,
                              right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = box.unpack(left, SIDE)
    right_psi, right_chi = box.unpack(right, SIDE)
    left_cross = dyad(left_chi, left_psi)
    right_cross = dyad(right_chi, right_psi)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        left_bond = left_cross-torch.roll(
            left_cross, shifts=offset, dims=(0, 1, 2))
        right_bond = right_cross-torch.roll(
            right_cross, shifts=offset, dims=(0, 1, 2))
        midpoint = (left_bond+right_bond)/2
        increment = right_bond-left_bond
        result += weight*torch.sum(torch.imag(
            torch.conj(midpoint)*increment))
    return result/(2*small.CHI0**2)


def local_gram_area_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    area = torch.linalg.cross(chi, psi, dim=-1)
    return torch.sum(torch.real(torch.sum(torch.conj(area)*area, dim=-1)))


def density_balance_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi)*chi, dim=-1))
    return torch.sum(rho_psi**2*(rho_chi-rho_psi)**2) / (
        4*small.CHI0**4)


def phase_rotate(raw: torch.Tensor, theta_psi: torch.Tensor,
                 theta_chi: torch.Tensor, sign: int) -> torch.Tensor:
    psi, chi = box.unpack(raw, SIDE)
    qp = torch.complex(torch.cos(theta_psi), sign * torch.sin(theta_psi))
    qc = torch.complex(torch.cos(theta_chi), sign * torch.sin(theta_chi))
    return torch.stack((torch.real(qp * psi), torch.imag(qp * psi),
                        torch.real(qc * chi), torch.imag(qc * chi)),
                       dim=-2).reshape(-1)


def interval(left: torch.Tensor, right: torch.Tensor,
             strength: float) -> torch.Tensor:
    energy = FRAME_ENERGY or complete_frame_energy
    face_left = sum(coefficient*face.energy(left, kind, part)
                    for coefficient, kind, part in BASE_FACE_TERMS)
    face_right = sum(coefficient*face.energy(right, kind, part)
                     for coefficient, kind, part in BASE_FACE_TERMS)
    return (box.interval(left, right, SIDE)
            + BASE_EVEN_STRENGTH*oscillator.oscillator_interval(
                left, right, "transfer_even")
            - BASE_DENSITY_STRENGTH*small.DT/2*(
                density_balance_energy(left)+density_balance_energy(right))
            - BASE_FACE_SCALE*small.DT/2*(face_left+face_right)
            + (strength*FRAME_INTERVAL(left, right)
               if FRAME_INTERVAL is not None else 0.0)
            - small.DT * strength / 2
            * (energy(left) + energy(right)))


def solve_strength(seed_parameters: np.ndarray, reference: np.ndarray,
                   strength: float) -> tuple[np.ndarray, list[dict]]:
    anchor = (0, 0, 0)
    exterior = (2, 2, 2)
    reference_psi, _reference_chi = small.unpack(reference)

    def residual_torch(parameters: torch.Tensor,
                       create_graph: bool) -> torch.Tensor:
        raw = parameters[:-2]
        theta_psi, theta_chi = parameters[-2], parameters[-1]
        previous = phase_rotate(raw, theta_psi, theta_chi, -1)
        following = phase_rotate(raw, theta_psi, theta_chi, 1)
        action = interval(previous, raw, strength) + interval(
            raw, following, strength)
        dynamic = torch.autograd.grad(
            action, raw, create_graph=create_graph)[0]
        psi, chi = box.unpack(raw, SIDE)
        gauge_psi = torch.imag(torch.sum(
            torch.conj(torch.tensor(reference_psi[anchor])) * psi[anchor]))
        gauge_chi = torch.imag(chi[exterior][2])
        return torch.cat((dynamic, gauge_psi[None], gauge_chi[None]))

    parameters = seed_parameters.copy()
    trace = []
    for iteration in range(15):
        point = torch.tensor(parameters, dtype=torch.float64,
                             requires_grad=True)
        residual = residual_torch(point, create_graph=True)
        value = residual.detach().numpy()
        rms = float(np.linalg.norm(value[:-2]) / np.sqrt(value.size - 2))
        total = float(np.linalg.norm(value))
        trace.append({"iteration": iteration, "DEL_rms": rms,
                      "augmented_norm": total})
        if rms < 1e-10 and np.linalg.norm(value[-2:]) < 1e-10:
            break
        jacobian = torch.autograd.functional.jacobian(
            lambda x: residual_torch(x, create_graph=True), point,
            vectorize=True).detach().numpy()
        accepted = False
        normal = jacobian.T @ jacobian
        rhs = -jacobian.T @ value
        diagonal_scale = max(float(np.max(np.diag(normal))), 1.0)
        # Global frame/phase symmetries make the bordered Jacobian singular.
        # A rank-revealing Gauss--Newton direction alone is unreliable here;
        # use a transparent Levenberg ladder and retain the first decrease.
        for damping_power in range(-16, 5):
            damping = 10.0**damping_power * diagonal_scale
            step = np.linalg.solve(
                normal + damping * np.eye(normal.shape[0]), rhs)
            for power in range(10):
                scale = 0.5**power
                trial = parameters + scale * step
                trial_tensor = torch.tensor(trial, dtype=torch.float64,
                                            requires_grad=True)
                trial_value = residual_torch(
                    trial_tensor, create_graph=False).detach().numpy()
                if np.linalg.norm(trial_value) < total:
                    parameters = trial
                    trace[-1]["accepted_scale"] = scale
                    trace[-1]["accepted_damping"] = damping
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            trace[-1]["accepted_scale"] = 0.0
            break
    return parameters, trace


def identity_metrics(reference: np.ndarray, state: np.ndarray) -> dict:
    psi0, _chi0 = small.unpack(reference)
    psi, chi = small.unpack(state)
    mask = np.sum(abs(psi0)**2, axis=-1) > 1e-12
    overlap = abs(np.vdot(psi0[mask], psi[mask]))**2 / (
        np.sum(abs(psi0[mask])**2) * np.sum(abs(psi[mask])**2))
    exterior = np.sum(abs(psi[~mask])**2) / max(np.sum(abs(psi)**2), 1e-300)
    vacuum = np.zeros_like(chi)
    vacuum[..., 2] = small.CHI0
    return {
        "Psi_carrier_fidelity": float(overlap),
        "Psi_exterior_fraction": float(exterior),
        "chi_exterior_deviation_rms": float(
            np.linalg.norm((chi-vacuum)[~mask])
            / np.sqrt(max(3*np.count_nonzero(~mask), 1))),
    }


def spectrum(parameters: np.ndarray, strength: float) -> dict:
    raw = parameters[:-2]
    theta_psi, theta_chi = parameters[-2:]
    current = torch.tensor(raw, dtype=torch.float64)
    tp, tc = [torch.tensor(value, dtype=torch.float64)
              for value in (theta_psi, theta_chi)]
    previous = phase_rotate(current, tp, tc, -1).detach()
    following = phase_rotate(current, tp, tc, 1).detach()

    def del_residual(p, c, f):
        return torch.func.grad(
            lambda x: interval(p, x, strength) + interval(x, f, strength))(c)

    layers = (previous, current, following)
    blocks = []
    for index in range(3):
        def function(value, selected=index):
            values = list(layers)
            values[selected] = value
            return del_residual(*values)
        blocks.append(torch.autograd.functional.jacobian(
            function, layers[index], vectorize=True).detach().numpy())
    a, b, c = blocks
    # Co-rotating return map for independently phased fields.
    dimension = raw.size
    basis = np.eye(dimension)
    rotation_columns = []
    for column in basis.T:
        rotation_columns.append(phase_rotate(
            torch.tensor(column), tp, tc, 1).detach().numpy())
    rotation = np.column_stack(rotation_columns)
    lhs = c @ rotation
    forward = -np.linalg.solve(lhs, b)
    backward = -np.linalg.solve(lhs, a @ rotation.T)
    identity = np.eye(dimension)
    monodromy = np.block([[forward, backward],
                          [identity, np.zeros_like(identity)]])
    values = eig(monodromy, right=False, check_finite=False)
    radius = float(np.max(abs(values)))
    return {
        "spectral_radius": radius,
        "growth_rate": float(np.log(radius) / small.DT),
        "count_above_1_plus_1e_minus_7": int(np.sum(abs(values) > 1+1e-7)),
    }


def main() -> None:
    global FRAME_ENERGY, FRAME_INTERVAL, FRAME_ENERGY_NAME
    global BASE_EVEN_STRENGTH, BASE_DENSITY_STRENGTH, BASE_FACE_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=(
        "complete", "ordered", "chi_orientation", "plane", "plane_kinetic", "area_berry",
        "cross_berry", "cross_oscillator", "cross_bond_oscillator",
        "cross_bond_berry",
        "local_area"),
                        default="complete")
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=list(STRENGTHS))
    parser.add_argument("--base-even-strength", type=float, default=0.0)
    parser.add_argument("--base-density-strength", type=float, default=0.0)
    parser.add_argument("--base-face-scale", type=float, default=0.0)
    args = parser.parse_args()
    FRAME_ENERGY = {
        "complete": complete_frame_energy,
        "ordered": ordered_frame_energy,
        "chi_orientation": chi_orientation_energy,
        "plane": plane_frame_energy,
        "plane_kinetic": zero_energy,
        "area_berry": zero_energy,
        "cross_berry": zero_energy,
        "cross_oscillator": zero_energy,
        "cross_bond_oscillator": zero_energy,
        "cross_bond_berry": zero_energy,
        "local_area": local_gram_area_energy,
    }[args.candidate]
    FRAME_INTERVAL = {
        "plane_kinetic": plane_kinetic_interval,
        "area_berry": area_berry_interval,
        "cross_berry": cross_berry_interval,
        "cross_oscillator": cross_oscillator_interval,
        "cross_bond_oscillator": cross_bond_oscillator_interval,
        "cross_bond_berry": cross_bond_berry_interval,
    }.get(args.candidate)
    FRAME_ENERGY_NAME = args.candidate
    BASE_EVEN_STRENGTH = args.base_even_strength
    BASE_DENSITY_STRENGTH = args.base_density_strength
    BASE_FACE_SCALE = args.base_face_scale
    face.SIDE = SIDE
    oscillator.large.SIDE = SIDE
    reference = np.load(
        HERE / "canonical_noncoplanar_solvability_search.npz")["rest"]
    theta = float(np.arccos(1 - small.DT**2 * small.CHI0**2 / 2))
    parameters = np.r_[reference, theta, 0.0]
    rows = []
    for strength in args.strengths:
        parameters, trace = solve_strength(
            parameters, reference, strength)
        raw = parameters[:-2]
        point = torch.tensor(parameters, dtype=torch.float64,
                             requires_grad=True)
        # Re-evaluate through a short solve with the same strength so the last
        # trace is the authoritative residual record.
        metrics = identity_metrics(reference, raw)
        row = {
            "strength": strength,
            "trace": trace,
            "theta_Psi": float(parameters[-2]),
            "theta_chi": float(parameters[-1]),
            **metrics,
        }
        if trace[-1]["DEL_rms"] < 1e-8:
            row.update(spectrum(parameters, strength))
        rows.append(row)
        if trace[-1]["DEL_rms"] >= 1e-6:
            break
    result = {
        "status": "V4_PERSONAL_TOUCH_COMPLETE_FRAME_BRANCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate": FRAME_ENERGY_NAME,
        "base_even_strength": BASE_EVEN_STRENGTH,
        "base_density_strength": BASE_DENSITY_STRENGTH,
        "base_face_scale": BASE_FACE_SCALE,
        "action_delta": {
            "complete": (
                "positive reciprocal C19 Hilbert--Schmidt differences of "
                "the complete ordered Psi/chi dyad tensor"),
            "ordered": (
                "positive reciprocal C19 Hilbert--Schmidt differences of "
                "the separate Psi and chi direction dyads"),
            "chi_orientation": (
                "positive reciprocal C19 Hilbert--Schmidt differences of "
                "the native chi direction dyad chi chi^dagger"),
            "plane": (
                "positive reciprocal C19 Hilbert--Schmidt differences of "
                "the joint local plane tensor Psi Psi^dagger + chi chi^dagger"),
            "plane_kinetic": (
                "positive local kinetic metric for the time derivative of "
                "the joint plane tensor Psi Psi^dagger + chi chi^dagger"),
            "area_berry": (
                "local reciprocal Berry one-form of the native oriented "
                "area W=chi wedge Psi"),
            "cross_berry": (
                "local reciprocal Berry one-form of the complete native "
                "cross-covariance tensor chi Psi^dagger"),
            "cross_oscillator": (
                "positive local harmonic action of the complete native "
                "cross-covariance tensor chi Psi^dagger, with relative "
                "kinetic/restoring weights fixed by the native chi0 clock"),
            "cross_bond_oscillator": (
                "positive C19 bond-harmonic action of differences of the "
                "complete native cross-covariance chi Psi^dagger, with "
                "relative weights fixed by C19 and the native chi0 clock"),
            "cross_bond_berry": (
                "C19 reciprocal Berry one-form of neighbor differences of "
                "the complete native cross-covariance chi Psi^dagger"),
            "local_area": (
                "positive local Gram-area energy |chi wedge Psi|^2"),
        }[args.candidate],
        "coefficient_rule": (
            "unit relative weights are fixed by the Hilbert--Schmidt norm; "
            "overall strength is continuation only and is not admitted"),
        "rows": rows,
        "verdict": (
            "COMPLETE_FRAME_BRANCH_IMPROVES_SPECTRUM"
            if any(row.get("growth_rate", np.inf) < 1.0360092627725035
                   for row in rows)
            else "COMPLETE_FRAME_BRANCH_SOLVE_INCONCLUSIVE"
            if not any("growth_rate" in row for row in rows)
            else "COMPLETE_FRAME_BRANCH_NO_IMPROVEMENT"),
        "claim_boundary": (
            "Unrestricted side-three rest-branch continuation and full dense "
            "spectrum. A favorable row still requires coefficient provenance, "
            "side-five/timestep refinement, nonlinear stability, and motion."),
    }
    suffix = FRAME_ENERGY_NAME
    if BASE_EVEN_STRENGTH:
        suffix += f"_even_{BASE_EVEN_STRENGTH:g}".replace(".", "p")
    if BASE_DENSITY_STRENGTH:
        suffix += f"_density_{BASE_DENSITY_STRENGTH:g}".replace(".", "p")
    if BASE_FACE_SCALE:
        suffix += f"_face_{BASE_FACE_SCALE:g}".replace(".", "p")
    output = HERE / f"personal_touch_{suffix}_frame_branch.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    np.savez_compressed(HERE / f"personal_touch_{suffix}_frame_branch.npz",
                        parameters=parameters)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
