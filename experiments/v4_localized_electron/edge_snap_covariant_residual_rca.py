"""Resolve a saved edge-handoff residual in the local Psi--chi frame.

Global complex component labels are not cubic-covariant. This diagnostic
splits each residual into the local chi ray, the additional direction in the
span of (chi,Psi), the Hermitian-orthogonal orientation direction, and sites
where the local frame is undefined. It is an RCA only; no projector enters
the action or evolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5
EPS = 1e-12


def local_split(residual: np.ndarray, psi: np.ndarray,
                chi: np.ndarray) -> dict[str, float]:
    total = float(np.vdot(residual, residual).real)
    if total == 0:
        return {"chi_ray": 0.0, "psi_plane": 0.0,
                "orientation_normal": 0.0, "unframed": 0.0}
    columns = []
    chi_norm = float(np.linalg.norm(chi))
    if chi_norm > EPS:
        columns.append(chi / chi_norm)
    psi_work = psi.copy()
    for basis in columns:
        psi_work -= basis * np.vdot(basis, psi_work)
    psi_norm = float(np.linalg.norm(psi_work))
    if psi_norm > EPS:
        columns.append(psi_work / psi_norm)
    if not columns:
        return {"chi_ray": 0.0, "psi_plane": 0.0,
                "orientation_normal": 0.0, "unframed": total}
    ray = abs(np.vdot(columns[0], residual))**2 if chi_norm > EPS else 0.0
    plane = 0.0
    if len(columns) == 2:
        plane = abs(np.vdot(columns[1], residual))**2
    spanned = ray + plane
    return {
        "chi_ray": float(ray),
        "psi_plane": float(plane),
        "orientation_normal": float(max(total - spanned, 0.0)),
        "unframed": 0.0,
    }


def main(input_stem: str, output_stem: str) -> None:
    data = np.load(HERE / f"{input_stem}.npz")
    states = [data["q0"], *list(data["middle"])]
    residuals = data["residuals"]
    names = ("chi_ray", "psi_plane", "orientation_normal", "unframed")
    aggregate = {
        "psi": {name: 0.0 for name in names},
        "chi": {name: 0.0 for name in names},
    }
    layer_rows = []
    site_rows = []
    for layer, (raw_state, raw_residual) in enumerate(zip(states, residuals)):
        psi, chi = rank.relative.unpack_raw(raw_state, SIDE)
        rpsi, rchi = rank.relative.unpack_raw(raw_residual, SIDE)
        layer_sums = {
            field: {name: 0.0 for name in names} for field in aggregate
        }
        for index in np.ndindex((SIDE, SIDE, SIDE)):
            for field, residual in (("psi", rpsi[index]),
                                    ("chi", rchi[index])):
                split = local_split(residual, psi[index], chi[index])
                for name, value in split.items():
                    aggregate[field][name] += value
                    layer_sums[field][name] += value
                site_rows.append({
                    "layer": layer,
                    "site": list(index),
                    "field": field,
                    "residual_squared": float(np.vdot(
                        residual, residual).real),
                    **split,
                })
        layer_rows.append({"layer": layer, "channels": layer_sums})
    total = sum(sum(channels.values()) for channels in aggregate.values())
    fractions = {
        field: {name: float(value / max(total, 1e-300))
                for name, value in channels.items()}
        for field, channels in aggregate.items()
    }
    largest = sorted(site_rows, key=lambda row: -row["residual_squared"])[:20]
    dominant = max(
        ((value, field, name) for field, channels in fractions.items()
         for name, value in channels.items()),
        key=lambda item: item[0])
    result = {
        "status": "V4_EDGE_HANDOFF_COVARIANT_RESIDUAL_RCA",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "input_stem": input_stem,
        "aggregate_fractions_of_total": fractions,
        "dominant_native_channel": {
            "fraction": dominant[0], "field": dominant[1],
            "channel": dominant[2]},
        "layers": layer_rows,
        "largest_sites": largest,
        "interpretation": (
            "chi_ray is amplitude/radial substrate response; psi_plane is "
            "in-plane orientation transfer; orientation_normal is the "
            "remaining complex C3 direction; unfamed sites lack a local "
            "carrier frame."),
        "claim_boundary": (
            "Measurement-only covariant RCA. The local frame is not an "
            "action term, runtime projector, or movement proof."),
    }
    (HERE / f"{output_stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-stem", default="edge_snap_transfer_charge_four_update_orbit")
    parser.add_argument(
        "--output-stem", default="edge_snap_transfer_covariant_residual_rca")
    args = parser.parse_args()
    main(args.input_stem, args.output_stem)
