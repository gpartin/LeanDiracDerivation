"""Compose a diagnostic strength scan of the matched-overlap oscillator."""

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
UNIT = HERE / (
    f"{PREFIX}_overlap_0_overlapface_0_overlapmatched_1_"
    "dt_0p00390625.npz")
SCALES = (6/19, 0.5, 1.0, 2.0, 3.0)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def main() -> None:
    names = ("previous", "current", "following")
    with np.load(BASE) as archive:
        base = {name: archive[name].copy() for name in names}
    with np.load(UNIT) as archive:
        unit = {name: archive[name].copy() for name in names}
    outputs = []
    for scale in SCALES:
        output = HERE / (
            f"{PREFIX}_overlap_0_overlapface_0_"
            f"overlapmatched_{tag(scale)}_dt_0p00390625.npz")
        blocks = {
            name: base[name] + scale*(unit[name]-base[name])
            for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"scale": scale, "output": output.name})
    report = {
        "status": "V4_MATCHED_OVERLAP_STRENGTH_SCAN_COMPOSED",
        "identity": "candidate = base + scale * (unit_matched - base)",
        "outputs": outputs,
    }
    (HERE / "compose_matched_overlap_strength_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
