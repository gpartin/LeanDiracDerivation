"""Compose the native rest-clock potential scan at unit material coupling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREFIX = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0")
BASE = HERE / f"{PREFIX}_dt_0p00390625.npz"
OVERLAP = HERE / "overlap_material_tangent_blocks.npz"
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"
UNIT = HERE / (
    f"{PREFIX}_overlap_0_overlapface_0_overlapmatched_1_"
    "dt_0p00390625.npz")
SCALES = (0.5, 1.0, 1.5, 2.0)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def load(path: Path, names: tuple[str, ...]):
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in names}


def main() -> None:
    names = ("previous", "current", "following")
    base = load(BASE, names)
    overlap = load(OVERLAP, names)
    incumbent = load(INCUMBENT, names)
    unit = load(UNIT, names)
    raw = {
        name: base[name]+overlap[name]-incumbent[name]
        for name in names
    }
    outputs = []
    for scale in SCALES:
        output = HERE / (
            f"{PREFIX}_overlap_0_overlapface_0_overlapmatched_1_"
            f"overlappotential_{tag(scale)}_dt_0p00390625.npz")
        blocks = {
            name: raw[name] + scale*(unit[name]-raw[name])
            for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"potential_scale": scale, "output": output.name})
    report = {
        "status": "V4_MATCHED_OVERLAP_POTENTIAL_SCAN_COMPOSED",
        "identity": "candidate = kinetic_only + p * matched_potential",
        "outputs": outputs,
    }
    (HERE / "compose_matched_overlap_potential_scan.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
