"""Rest-preserving reciprocal oscillator completion of the transfer wake.

For each directed C19 bond define

    z_ij = chi_i^dagger (chi_j wedge Psi_j),
    Z_ij^+/- = z_ij +/- z_ji.

The exact rest carrier has Z_(n+1)=exp(i theta) Z_n.  The discrete harmonic
action with omega=chi0 has this sequence as an exact DEL solution because
2(1-cos theta)=dt^2 chi0^2.  This script tests whether the same action keeps
the robust fixed-charge transport response found in the kinetic-only screen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_wake_kinetic_gate as gate
import localized_stability as native


HERE = Path(__file__).resolve().parent


def transfer_coordinates(raw: torch.Tensor, mode: str) -> list[tuple[int, torch.Tensor]]:
    psi, chi = gate.box.unpack(raw, large.SIDE)
    w = torch.linalg.cross(chi, psi, dim=-1)
    values = []
    for offset, weight in native.DIRECTIONS:
        chi_j = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        w_j = torch.roll(w, shifts=offset, dims=(0, 1, 2))
        z_ij = torch.sum(torch.conj(chi)*w_j, dim=-1)
        z_ji = torch.sum(torch.conj(chi_j)*w, dim=-1)
        z = z_ij+z_ji if mode == "transfer_even" else z_ij-z_ji
        values.append((weight, z))
    return values


def oscillator_interval(left: torch.Tensor, right: torch.Tensor,
                        mode: str, dt: float | None = None) -> torch.Tensor:
    if dt is None:
        dt = small.DT
    left_values = transfer_coordinates(left, mode)
    right_values = transfer_coordinates(right, mode)
    kinetic = torch.zeros((), dtype=torch.float64)
    potential = torch.zeros((), dtype=torch.float64)
    for (weight, zl), (_weight, zr) in zip(
            left_values, right_values, strict=True):
        difference = zr-zl
        kinetic += weight*torch.sum(torch.real(
            torch.conj(difference)*difference))
        potential += weight*torch.sum(
            torch.real(torch.conj(zl)*zl+torch.conj(zr)*zr))
    normalization = small.CHI0**4
    return (kinetic/(4*dt*normalization)
            - dt*small.CHI0**2*potential/(8*normalization))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(
        "transfer_even", "transfer_odd"), default="transfer_odd")
    parser.add_argument("--strength", type=float, default=1e-4)
    args = parser.parse_args()
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    old = gate.extra_interval
    gate.extra_interval = oscillator_interval
    try:
        selected = [(0.0, archive["rest"]),
                    (float(archive["fractions"][-1]),
                     archive["states"][-1])]
        rows = [gate.run(args.mode, args.strength, state, fraction, phase)
                for fraction, state in selected]
    finally:
        gate.extra_interval = old
    rest_pass = all(row["candidate_rest_DEL_rms"] < 1e-9 for row in rows)
    rank_pass = all(row["axis_ranks"] == [3, 3, 3]
                    and row["minimum_relative_third_singular_value"] > 1e-6
                    for row in rows)
    exact_frequency_identity = float(
        abs(2*(1-np.cos(theta))-small.DT**2*small.CHI0**2))
    result = {
        "status": "V4_CANONICAL_TRANSFER_OSCILLATOR_GATE",
        "mode": args.mode,
        "strength": args.strength,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "positive reciprocal discrete harmonic action for "
            "Z_ij=chi_i^dagger(chi_j wedge Psi_j) +/- reverse bond"),
        "coefficient_provenance": (
            "relative kinetic/potential coefficient fixed by the native "
            "rest identity 2(1-cos theta)=dt^2 chi0^2; overall strength "
            "remains unselected"),
        "frequency_identity_error": exact_frequency_identity,
        "rows": rows,
        "rest_gate_pass": rest_pass,
        "transport_gate_pass": rank_pass,
        "verdict": (
            "TRANSFER_OSCILLATOR_REST_AND_RANK_PASS"
            if rest_pass and rank_pass else
            "TRANSFER_OSCILLATOR_GATE_FAIL"),
        "claim_boundary": (
            "Exact rest DEL and linear fixed-charge rank only. A pass still "
            "requires positive Hamiltonian/time-update proof, coefficient "
            "selection, spectrum, Noether-current equality, continuum audit, "
            "and unrestricted finite translation."),
    }
    suffix = f"{args.strength:g}".replace("-", "m").replace(".", "p")
    output = HERE / f"canonical_transfer_oscillator_{args.mode}_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "frequency_identity_error": exact_frequency_identity,
        "rows": [{
            "fraction": row["fraction_of_rest_norm"],
            "rest_rms": row["candidate_rest_DEL_rms"],
            "added_rest_rms": row["added_rest_DEL_rms"],
            "condition": row["previous_block_condition_number"],
            "ranks": row["axis_ranks"],
            "minimum_relative_third": row[
                "minimum_relative_third_singular_value"]}
            for row in rows],
        "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
