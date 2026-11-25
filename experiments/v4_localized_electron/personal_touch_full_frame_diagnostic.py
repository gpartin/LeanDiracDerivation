"""Resolve the canonical and repaired soft modes into complete Psi--chi data.

This is a measurement script.  It does not add a force or select an action.
Every reported derivative is a JVP of a native local or C19 observable at the
same exact localized rest orbit used by the side-five Floquet calculation.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_oscillator_gate as oscillator


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE
CORE = tuple(product((1, 2), repeat=3))


def real_coordinates(value: torch.Tensor) -> torch.Tensor:
    """Flatten a real or complex tensor into real coordinates."""
    if torch.is_complex(value):
        return torch.view_as_real(value).reshape(-1)
    return value.reshape(-1)


def core_values(field: torch.Tensor) -> torch.Tensor:
    return torch.stack([field[index] for index in CORE])


def normalized(field: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(torch.real(torch.sum(torch.conj(field) * field, dim=-1)))
    safe = torch.where(norm > 1e-14, norm, torch.ones_like(norm))
    unit = field / safe[..., None]
    return torch.where((norm > 1e-14)[..., None], unit, torch.zeros_like(unit))


def configuration_observables(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    psi, chi = box.unpack(raw, SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi) * psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi) * chi, dim=-1))
    area = torch.linalg.cross(chi, psi, dim=-1)
    overlap = torch.sum(torch.conj(chi) * psi, dim=-1)
    upsi, uchi, uarea = normalized(psi), normalized(chi), normalized(area)
    frame = torch.stack((upsi, uchi, uarea), dim=-1)

    transfers = []
    for site in CORE:
        fi = frame[site]
        for offset, _weight in oscillator.native.DIRECTIONS:
            neighbor = tuple((site[axis] + offset[axis]) % SIDE
                             for axis in range(3))
            fj = frame[neighbor]
            transfers.append(torch.conj(fi).transpose(-2, -1) @ fj)
    transfer = torch.stack(transfers)

    return {
        "rho_Psi": core_values(rho_psi),
        "rho_chi": core_values(rho_chi),
        "wedge_chi_cross_Psi": core_values(area),
        "parallel_overlap_chi_dagger_Psi": core_values(overlap),
        "unit_Psi_direction": core_values(upsi),
        "unit_chi_direction": core_values(uchi),
        "unit_joint_normal": core_values(uarea),
        "complete_local_tripod": core_values(frame),
        "C19_neighbor_tripod_transfer": transfer,
    }


def phase_space_observables(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    dimension = raw.numel() // 2
    previous, current = raw[:dimension], raw[dimension:]
    psi_prev, chi_prev = box.unpack(previous, SIDE)
    psi, chi = box.unpack(current, SIDE)
    psi_dot = (psi - psi_prev) / small.DT
    chi_dot = (chi - chi_prev) / small.DT

    def phase_step(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        numerator = torch.sum(torch.conj(left) * right, dim=-1)
        left_norm = torch.sqrt(torch.real(torch.sum(
            torch.conj(left) * left, dim=-1)))
        right_norm = torch.sqrt(torch.real(torch.sum(
            torch.conj(right) * right, dim=-1)))
        denominator = left_norm * right_norm
        safe = torch.where(denominator > 1e-14, denominator,
                           torch.ones_like(denominator))
        value = numerator / safe
        return torch.where(denominator > 1e-14, value,
                           torch.zeros_like(value))

    psi_step = phase_step(psi_prev, psi)
    chi_step = phase_step(chi_prev, chi)
    relative_step = psi_step * torch.conj(chi_step)
    relative_velocity = (
        torch.linalg.cross(chi, psi_dot, dim=-1)
        - torch.linalg.cross(chi_dot, psi, dim=-1))
    common_area_velocity = (
        torch.linalg.cross(chi_dot, psi, dim=-1)
        + torch.linalg.cross(chi, psi_dot, dim=-1))
    phase_rate_psi = torch.imag(torch.sum(torch.conj(psi) * psi_dot, dim=-1))
    phase_rate_chi = torch.imag(torch.sum(torch.conj(chi) * chi_dot, dim=-1))
    return {
        "Psi_layer_phase_step": core_values(psi_step),
        "chi_layer_phase_step": core_values(chi_step),
        "relative_Psi_chi_clock_step": core_values(relative_step),
        "relative_frame_velocity": core_values(relative_velocity),
        "common_area_velocity": core_values(common_area_velocity),
        "Psi_phase_rate": core_values(phase_rate_psi),
        "chi_phase_rate": core_values(phase_rate_chi),
        "relative_phase_rate": core_values(phase_rate_psi - phase_rate_chi),
    }


def jvp_table(function, point: torch.Tensor,
              direction: np.ndarray) -> dict[str, dict[str, float]]:
    direction = np.asarray(direction)
    real_direction = torch.tensor(np.real(direction), dtype=torch.float64)
    imag_direction = torch.tensor(np.imag(direction), dtype=torch.float64)
    base = function(point)
    state_norm = float(torch.linalg.vector_norm(point))
    direction_norm = float(np.linalg.norm(direction))
    rows = {}
    for name in base:
        def component(value, key=name):
            return real_coordinates(function(value)[key])
        output, derivative_real = torch.autograd.functional.jvp(
            component, point, real_direction, create_graph=False)
        _output, derivative_imag = torch.autograd.functional.jvp(
            component, point, imag_direction, create_graph=False)
        derivative = (derivative_real.detach().numpy()
                      + 1j * derivative_imag.detach().numpy())
        output_norm = float(torch.linalg.vector_norm(output).detach())
        derivative_norm = float(np.linalg.norm(derivative))
        relative_input = direction_norm / max(state_norm, 1e-300)
        rows[name] = {
            "coordinate_count": int(output.numel()),
            "rest_output_norm": output_norm,
            "jvp_norm": derivative_norm,
            "jvp_rms": derivative_norm / np.sqrt(max(output.numel(), 1)),
            "dimensionless_relative_sensitivity": (
                derivative_norm / output_norm / relative_input
                if output_norm > 1e-12 and relative_input > 0 else None),
        }
    return rows


def audit_mode(label: str, archive_name: str) -> dict:
    archive = np.load(HERE / archive_name)
    rest = archive["rest"]
    vector = archive["phase_space_vector"]
    dimension = rest.size
    current_mode, previous_mode = vector[:dimension], vector[dimension:]
    theta = float(np.arccos(1 - small.DT**2 * small.CHI0**2 / 2))
    current = torch.tensor(rest, dtype=torch.float64)
    previous = box.phase_rotate(
        current, np.exp(-1j * theta), SIDE).detach()
    phase_point = torch.cat((previous, current))
    phase_direction = np.r_[previous_mode, current_mode]
    return {
        "label": label,
        "source_archive": archive_name,
        "eigenvalue": {
            "real": float(np.real(archive["eigenvalue"])),
            "imag": float(np.imag(archive["eigenvalue"])),
            "modulus": float(np.abs(archive["eigenvalue"])),
        },
        "configuration_observable_jvps": jvp_table(
            configuration_observables, current, current_mode),
        "phase_space_observable_jvps": jvp_table(
            phase_space_observables, phase_point, phase_direction),
    }


def main() -> None:
    modes = [
        audit_mode(
            "canonical_V40",
            "canonical_transfer_oscillator_spectrum_0_leading_mode.npz"),
        audit_mode(
            "best_exact_rest_diagnostic",
            "canonical_transfer_oscillator_spectrum_0_even_2_density_0p5_leading_mode.npz"),
    ]
    result = {
        "status": "V4_PERSONAL_TOUCH_FULL_FRAME_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "modes": modes,
        "verdict": "FULL_FRAME_SOFT_MODE_MEASURED",
        "claim_boundary": (
            "Directional derivatives of declared observables at the exact rest "
            "orbit. This diagnoses action sensitivity; it is not an action "
            "repair, stability pass, or movement pass."),
    }
    output = HERE / "personal_touch_full_frame_diagnostic.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
