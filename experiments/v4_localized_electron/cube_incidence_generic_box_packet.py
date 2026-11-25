"""Larger-box native action residual of the extended Dirac packet."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_extended_packet_gate as side5_packet
import cube_incidence_generic_action as action
import localized_stability as native


HERE = Path(__file__).resolve().parent


def centered_offsets(side):
    radius = side//2
    return list(itertools.product(range(-radius, radius+1), repeat=3))


def carrier_frames(rest, side):
    psi, chi = native.unpack(rest, side)
    sites = [tuple(int(x) for x in site)
             for site in np.argwhere(native.norm2(psi) > 1e-12)]
    sites.sort()
    pv = np.stack([psi[site] for site in sites])
    cv = np.stack([chi[site] for site in sites])
    return (pv/np.linalg.norm(pv, axis=1)[:, None],
            cv/np.linalg.norm(cv, axis=1)[:, None], sites)


def synthesize(field, basis, psi_frame, chi_frame, sites, offsets,
               chi_scale, side):
    dpsi = np.zeros((side, side, side, 3), dtype=complex)
    dchi = np.zeros_like(dpsi)
    flat = field.reshape(-1, 4)
    for center, offset in enumerate(offsets):
        corner_amplitudes = basis@flat[center]
        for corner, site in enumerate(sites):
            target = tuple((site[axis]+offset[axis]) % side
                           for axis in range(3))
            dpsi[target] += corner_amplitudes[corner]*psi_frame[corner]
            dchi[target] += (chi_scale*corner_amplitudes[corner]
                             * chi_frame[corner])
    return native_box.pack(dpsi, dchi)


def dirac_step(field, alpha, beta, sign, dt):
    side = field.shape[0]
    fourier = np.fft.fftn(field, axes=(0, 1, 2))
    result = np.empty_like(fourier)
    frequencies = 2*np.pi*np.fft.fftfreq(side)
    for index in itertools.product(range(side), repeat=3):
        momentum = np.asarray([frequencies[v] for v in index])
        transport = side5_packet.c19_first_symbol(momentum)
        hamiltonian = native_time.CHI0*(beta-np.eye(4))
        for axis in range(3):
            hamiltonian += transport[axis]*alpha[axis]
        result[index] = expm(-1j*sign*dt*hamiltonian)@fourier[index]
    return np.fft.ifftn(result, axes=(0, 1, 2))


def centered_dirac_step(field, alpha, beta, sign, dt):
    """Exact centered-time Dirac phase for an arbitrary periodic side."""
    side = field.shape[0]
    fourier = np.fft.fftn(field, axes=(0, 1, 2))
    result = np.empty_like(fourier)
    frequencies = 2*np.pi*np.fft.fftfreq(side)
    theta0 = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    for index in itertools.product(range(side), repeat=3):
        momentum = np.asarray([frequencies[v] for v in index])
        transport = side5_packet.c19_first_symbol(momentum)
        hamiltonian = native_time.CHI0*beta
        for axis in range(3):
            hamiltonian += transport[axis]*alpha[axis]
        energy = float(np.sqrt(native_time.CHI0**2+transport@transport))
        theta = float(np.arccos(np.clip(1-dt**2*energy**2/2, -1, 1)))
        lab = (np.cos(theta)*np.eye(4)
               - 1j*sign*np.sin(theta)*hamiltonian/energy)
        result[index] = np.exp(1j*sign*theta0)*lab@fourier[index]
    return np.fft.ifftn(result, axes=(0, 1, 2))


def packet(side, axis, sign, alpha, beta, dt, width=1.35, wave_number=0.35,
           centered_time=False):
    offsets = centered_offsets(side)
    coordinates = np.asarray(offsets, dtype=float).reshape(side, side, side, 3)
    distance2 = np.sum(coordinates**2, axis=-1)
    momentum = np.zeros(3)
    if axis == 3:
        momentum[:] = sign*wave_number/np.sqrt(3)
    else:
        momentum[axis] = sign*wave_number
    envelope = np.exp(-distance2/(2*width**2))*np.exp(
        1j*np.einsum("...a,a->...", coordinates, momentum))
    h0 = native_time.CHI0*beta
    transport = side5_packet.c19_first_symbol(momentum)
    for component in range(3):
        h0 += transport[component]*alpha[component]
    values, vectors = np.linalg.eigh(h0)
    current = envelope[..., None]*vectors[:, np.argmax(values)]
    stepper = centered_dirac_step if centered_time else dirac_step
    return (stepper(current, alpha, beta, -1, dt), current,
            stepper(current, alpha, beta, 1, dt), offsets)


def tangent_images(rest_layers, directions, side):
    images = []
    for selected in range(3):
        def function(value, index=selected):
            layers = list(rest_layers)
            layers[index] = value
            return action.del_residual(*layers, side)
        _value, image = torch.autograd.functional.jvp(
            function, rest_layers[selected], directions[selected],
            create_graph=False, strict=False)
        images.append(image.detach().numpy())
    return images


def main():
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    phase = np.exp(1j*theta)
    rows = []
    for side in (5, 7, 9):
        psi, chi = native.carrier_fields(side, native_time.AMPLITUDE_SQ)
        rest = native_box.pack(psi, chi)
        rest_current = torch.tensor(rest, dtype=torch.float64,
                                    requires_grad=True)
        rest_layers = (
            native_box.phase_rotate(rest_current.detach(), np.conjugate(phase), side),
            rest_current,
            native_box.phase_rotate(rest_current.detach(), phase, side),
        )
        psi_frame, chi_frame, sites = carrier_frames(rest, side)
        for flavor, basis in side5_packet.flavor_bases().items():
            alpha, beta = side5_packet.compressed_clifford(basis)
            for axis, sign in itertools.product((0, 3), (-1, 1)):
                previous, current, following, offsets = packet(
                    side, axis, sign, alpha, beta, native_time.DT)
                co_layers = tuple(synthesize(
                    field, basis, psi_frame, chi_frame, sites, offsets,
                    scales[flavor], side)
                    for field in (previous, current, following))
                directions = (
                    native_box.phase_rotate(torch.tensor(co_layers[0]),
                                            np.conjugate(phase), side),
                    torch.tensor(co_layers[1], dtype=torch.float64),
                    native_box.phase_rotate(torch.tensor(co_layers[2]),
                                            phase, side),
                )
                images = tangent_images(rest_layers, directions, side)
                residual = sum(images)
                scale = sum(np.linalg.norm(image) for image in images)
                rows.append({
                    "side": side,
                    "flavor": flavor,
                    "direction": "diagonal" if axis == 3 else "x",
                    "sign": sign,
                    "normalized_native_DEL_residual": float(
                        np.linalg.norm(residual)/max(scale, 1e-30)),
                    "absolute_native_DEL_RMS": float(
                        np.linalg.norm(residual)/np.sqrt(residual.size)),
                })
    per_side = []
    for side in (5, 7, 9):
        selected = [row for row in rows if row["side"] == side]
        per_side.append({
            "side": side,
            "maximum_normalized_native_DEL_residual": max(
                row["normalized_native_DEL_residual"] for row in selected),
            "maximum_absolute_native_DEL_RMS": max(
                row["absolute_native_DEL_RMS"] for row in selected),
        })
    result = {
        "status": "V4_CUBE_INCIDENCE_GENERIC_BOX_PACKET",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "rows": rows,
        "per_side": per_side,
        "verdict": ("GENERIC_BOX_PACKET_RESIDUAL_CONVERGES"
                    if per_side[-1]["maximum_normalized_native_DEL_residual"]
                    < per_side[0]["maximum_normalized_native_DEL_residual"]
                    else "GENERIC_BOX_PACKET_RESIDUAL_OPEN"),
        "claim_boundary": (
            "Full generic-action tangent DEL residual for the independent "
            "Dirac packet. It is not a solved nonlinear update or V4-006."),
    }
    (HERE/"cube_incidence_generic_box_packet.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
