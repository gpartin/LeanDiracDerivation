"""Box-refinement DEL gate for the live six-face action and Dirac packet."""

from __future__ import annotations

import itertools
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_generic_box_packet as generic_packet
import live_clifford_bond_action as side5_action
import live_clifford_generic_action as action
import localized_stability as native


HERE = Path(__file__).resolve().parent


def tangent_images(rest_layers, directions, side):
    images = []
    for selected in range(3):
        def function(value, index=selected):
            layers = list(rest_layers)
            layers[index] = value
            return action.del_residual(
                *layers, native_time.DT, side)
        _value, image = torch.autograd.functional.jvp(
            function, rest_layers[selected], directions[selected],
            create_graph=False, strict=False)
        images.append(image.detach().numpy())
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", nargs="+", type=int, default=(5, 7, 9))
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--two-wavelengths", action="store_true")
    parser.add_argument("--output", default="live_face_generic_box_gate.json")
    args = parser.parse_args()
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    phase = np.exp(1j*theta)
    rows = []
    side5_identity = None
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    chi_scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        chi_scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    for side in args.sides:
        psi, chi = native.carrier_fields(side, native_time.AMPLITUDE_SQ)
        rest = native_box.pack(psi, chi)
        current = torch.tensor(rest, dtype=torch.float64, requires_grad=True)
        rest_layers = (
            native_box.phase_rotate(
                current.detach(), np.conjugate(phase), side),
            current,
            native_box.phase_rotate(current.detach(), phase, side),
        )
        rest_residual = action.del_residual(
            *rest_layers, native_time.DT, side).detach().numpy()
        if side == 5:
            old = side5_action.del_residual(
                *rest_layers, native_time.DT).detach().numpy()
            side5_identity = {
                "DEL_absolute_difference": float(np.linalg.norm(old-rest_residual)),
                "DEL_relative_difference": float(
                    np.linalg.norm(old-rest_residual)
                    / max(np.linalg.norm(old), 1e-30)),
            }
        psi_frame, chi_frame, sites = generic_packet.carrier_frames(rest, side)
        offsets = generic_packet.centered_offsets(side)
        specifications = list(itertools.product(
            dirac.flavor_bases().items(), (0, 3), (-1, 1)))[
                :args.max_cases]
        for (flavor, basis), axis, sign in specifications:
            alpha, beta = dirac.compressed_clifford(basis)
            # Freeze the previously derived local lift across every box; it
            # is not reoptimized against the larger-box result.
            chi_scale = chi_scales[flavor]
            wave_number = (4*np.pi/side if args.two_wavelengths else 0.35)
            width = (side/8 if args.two_wavelengths else 1.35)
            previous, packet_current, following, _ = generic_packet.packet(
                side, axis, sign, alpha, beta, native_time.DT,
                width=width, wave_number=wave_number, centered_time=True)
            co_layers = tuple(generic_packet.synthesize(
                field, basis, psi_frame, chi_frame, sites, offsets,
                chi_scale, side)
                for field in (previous, packet_current, following))
            directions = (
                native_box.phase_rotate(
                    torch.tensor(co_layers[0]), np.conjugate(phase), side),
                torch.tensor(co_layers[1], dtype=torch.float64),
                native_box.phase_rotate(
                    torch.tensor(co_layers[2]), phase, side),
            )
            images = tangent_images(rest_layers, directions, side)
            residual = sum(images)
            normalization = sum(np.linalg.norm(image) for image in images)
            rows.append({
                "side": side,
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "x",
                "sign": sign,
                "packet_width": width,
                "wave_number": wave_number,
                "wavelengths_in_box": side*wave_number/(2*np.pi),
                "normalized_DEL_residual": float(
                    np.linalg.norm(residual)/max(normalization, 1e-30)),
                "absolute_DEL_RMS": float(
                    np.linalg.norm(residual)/np.sqrt(residual.size)),
                "rest_DEL_RMS": float(
                    np.linalg.norm(rest_residual)/np.sqrt(rest_residual.size)),
            })
    per_side = []
    for side in args.sides:
        selected = [row for row in rows if row["side"] == side]
        per_side.append({
            "side": side,
            "maximum_normalized_DEL_residual": max(
                row["normalized_DEL_residual"] for row in selected),
            "maximum_absolute_DEL_RMS": max(
                row["absolute_DEL_RMS"] for row in selected),
            "maximum_rest_DEL_RMS": max(
                row["rest_DEL_RMS"] for row in selected),
        })
    result = {
        "status": "V4_LIVE_FACE_GENERIC_BOX_GATE",
        "canonical_action_changed": False,
        "research_action_changed": True,
        "side5_identity": side5_identity,
        "rows": rows,
        "per_side": per_side,
        "verdict": (
            "LIVE_FACE_GENERIC_BOX_STABILITY_PASS"
            if (5 not in args.sides
                or side5_identity["DEL_absolute_difference"] < 1e-12)
            and max(row["maximum_rest_DEL_RMS"] for row in per_side) < 1e-9
            and max(row["maximum_normalized_DEL_residual"]
                    for row in per_side) < 2.5e-4
            and all(right["maximum_absolute_DEL_RMS"]
                    <= left["maximum_absolute_DEL_RMS"]
                    for left, right in zip(per_side, per_side[1:]))
            else "LIVE_FACE_GENERIC_BOX_OPEN"),
        "claim_boundary": (
            "V4-002 box check: exact side-five action identity, retained "
            "rest orbit, and a uniformly bounded physical packet response "
            "whose absolute RMS does not worsen on larger boxes. Vanishing "
            "Dirac transport remainder and unrestricted cell handoff remain "
            "V4-005/V4-006 gates."),
    }
    result["two_wavelength_scaling"] = args.two_wavelengths
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
