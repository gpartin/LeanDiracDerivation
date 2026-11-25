"""Trace the canonical C19 wake of the exact rotating carrier.

The localized carrier rotates by one common complex phase.  This audit asks
whether its native area bonds create a directional receiving-site wake.  It
keeps each occupied-neighbor contribution separate before summing it at the
four sites immediately beyond each cube face.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_large_box_audit as large
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def bond_receiver_force(psi_r: np.ndarray, chi_r: np.ndarray,
                        psi_s: np.ndarray, chi_s: np.ndarray,
                        weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Force at receiver from one undirected canonical area bond."""
    raw = torch.tensor(np.stack((psi_r.real, psi_r.imag,
                                 chi_r.real, chi_r.imag)),
                       dtype=torch.float64, requires_grad=True)
    sender_psi = torch.tensor(psi_s, dtype=torch.complex128)
    sender_chi = torch.tensor(chi_s, dtype=torch.complex128)
    receiver_psi = torch.complex(raw[0], raw[1])
    receiver_chi = torch.complex(raw[2], raw[3])
    wr = torch.linalg.cross(receiver_chi, receiver_psi)
    ws = torch.linalg.cross(sender_chi, sender_psi)
    # The directed C19 action contains both orientations.  Together they are
    # w |W_s-W_r|^2 / chi0^2 for one undirected bond.
    energy = weight*torch.real(torch.vdot(ws-wr, ws-wr))/small.CHI0**2
    gradient = torch.autograd.grad(energy, raw)[0].detach().numpy()
    return (-(gradient[0]+1j*gradient[1]),
            -(gradient[2]+1j*gradient[3])/small.B)


def receiving_sites(axis: int, sign: int):
    origin = SIDE//2-1
    coordinate = origin+2 if sign > 0 else origin-1
    sites = []
    for transverse in itertools.product((origin, origin+1), repeat=2):
        site = [0, 0, 0]
        site[axis] = coordinate
        cursor = 0
        for other in range(3):
            if other != axis:
                site[other] = transverse[cursor]
                cursor += 1
        sites.append(tuple(site))
    return sites


def analyze(state: np.ndarray, fraction: float) -> dict:
    psi, chi = large.unpack_np(state)
    psi_norm = np.linalg.norm(psi, axis=-1)
    occupied = psi_norm > 1e-10
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            site_rows = []
            for receiver in receiving_sites(axis, sign):
                contributions = []
                for offset, weight in native.DIRECTIONS:
                    sender = tuple((receiver[d]+offset[d]) % SIDE
                                   for d in range(3))
                    if not occupied[sender]:
                        continue
                    fp, fc = bond_receiver_force(
                        psi[receiver], chi[receiver],
                        psi[sender], chi[sender], weight)
                    contributions.append((offset, weight, sender, fp, fc))
                psi_forces = [item[3] for item in contributions]
                chi_forces = [item[4] for item in contributions]
                total_psi = np.sum(psi_forces, axis=0) if psi_forces \
                    else np.zeros(3, dtype=complex)
                total_chi = np.sum(chi_forces, axis=0) if chi_forces \
                    else np.zeros(3, dtype=complex)
                absolute_psi = float(sum(np.linalg.norm(value)
                                         for value in psi_forces))
                # With W_receiver=0, the native bond phase current is zero
                # on every incoming bond, independently of sender phase.
                phase_currents = []
                wr = np.cross(chi[receiver], psi[receiver])
                for _offset, weight, sender, _fp, _fc in contributions:
                    ws = np.cross(chi[sender], psi[sender])
                    phase_currents.append(float(
                        2*weight/small.CHI0**2
                        * np.imag(np.vdot(wr, ws))))
                site_rows.append({
                    "receiver": list(receiver),
                    "occupied_sender_count": len(contributions),
                    "individual_psi_force_norms": [
                        float(np.linalg.norm(value)) for value in psi_forces],
                    "sum_individual_psi_force_norms": absolute_psi,
                    "net_psi_force_norm": float(np.linalg.norm(total_psi)),
                    "force_cancellation_ratio": float(
                        np.linalg.norm(total_psi)/max(absolute_psi, 1e-300)),
                    "net_chi_acceleration_norm": float(
                        np.linalg.norm(total_chi)),
                    "incoming_native_phase_currents": phase_currents,
                    "net_incoming_native_phase_current": float(
                        sum(phase_currents)),
                })
            rows.append({
                "axis": axis,
                "sign": sign,
                "sites": site_rows,
                "maximum_net_psi_force_norm": float(max(
                    row["net_psi_force_norm"] for row in site_rows)),
                "maximum_force_cancellation_ratio": float(max(
                    row["force_cancellation_ratio"] for row in site_rows)),
                "maximum_native_phase_current": float(max(
                    abs(row["net_incoming_native_phase_current"])
                    for row in site_rows)),
            })
    return {"fraction_of_rest_norm": fraction, "face_rows": rows}


def main() -> None:
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    states = [(0.0, archive["rest"]),
              (float(archive["fractions"][-1]), archive["states"][-1])]
    analyses = [analyze(state, fraction) for fraction, state in states]
    maximum_force = max(row["maximum_net_psi_force_norm"]
                        for item in analyses for row in item["face_rows"])
    maximum_current = max(row["maximum_native_phase_current"]
                          for item in analyses for row in item["face_rows"])
    maximum_ratio = max(row["maximum_force_cancellation_ratio"]
                        for item in analyses for row in item["face_rows"])
    result = {
        "status": "V4_CANONICAL_ROTOR_WAKE_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "analyses": analyses,
        "maximum_receiving_psi_force_norm": maximum_force,
        "maximum_receiving_phase_current": maximum_current,
        "maximum_force_cancellation_ratio": maximum_ratio,
        "verdict": (
            "COMMON_ROTATION_HAS_DIRECTIONAL_NATIVE_WAKE"
            if maximum_force > 1e-9 or maximum_current > 1e-9 else
            "COMMON_ROTATION_WAKE_CANCELS_AT_RECEIVING_SITES"),
        "mechanism": (
            "Each incoming C19 bond exerts a nonzero Psi force, but the cube "
            "corner contributions cancel at every adjacent receiving site. "
            "A common temporal phase multiplies all contributions equally, "
            "so it preserves the cancellation and creates no spatial phase "
            "lag or Noether current."),
        "claim_boundary": (
            "Exact first-shell bond decomposition on the canonical rest orbit "
            "and exact side-five chi branch. This is not a nonlinear motion "
            "test or a no-go for every reciprocal action."),
    }
    output = HERE / "canonical_rotor_wake_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "maximum_receiving_psi_force_norm": maximum_force,
        "maximum_receiving_phase_current": maximum_current,
        "maximum_force_cancellation_ratio": maximum_ratio,
        "samples": [{
            "fraction": item["fraction_of_rest_norm"],
            "faces": [{
                "axis": row["axis"], "sign": row["sign"],
                "max_net_force": row["maximum_net_psi_force_norm"],
                "max_cancellation": row[
                    "maximum_force_cancellation_ratio"]}
                for row in item["face_rows"]]}
            for item in analyses],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
