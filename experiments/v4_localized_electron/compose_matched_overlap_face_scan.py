"""Compose geometric face strengths on the matched-overlap incumbent."""

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
FACE = HERE / f"{PREFIX}_overlap_0_overlapface_1_dt_0p00390625.npz"
MATCHED = HERE / (
    f"{PREFIX}_overlap_0_overlapface_0_overlapmatched_1_"
    "dt_0p00390625.npz")
SCALES = (1/19, 6/19, 1.0)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def main() -> None:
    base = np.load(BASE)
    face = np.load(FACE)
    matched = np.load(MATCHED)
    names = ("previous", "current", "following")
    outputs = []
    for scale in SCALES:
        output = HERE / (
            f"{PREFIX}_overlap_0_overlapface_{tag(scale)}_"
            "overlapmatched_1_dt_0p00390625.npz")
        blocks = {
            name: matched[name] + scale*(face[name]-base[name])
            for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"scale": scale, "output": output.name})
    report = {
        "status": "V4_MATCHED_OVERLAP_FACE_SCAN_COMPOSED",
        "identity": "candidate = matched + scale * (unit_face - base)",
        "outputs": outputs,
    }
    (HERE / "compose_matched_overlap_face_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
