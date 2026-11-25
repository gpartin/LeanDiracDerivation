"""Compose exact material-area tangent strengths from scale zero and one."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


HERE = Path(__file__).resolve().parent
HEAD = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p016620498615_symplectic_0_overlap_0_"
    "overlapface_0_overlapmatched_1_overlappotential_1_triangle_0")
ZERO = HERE / f"{HEAD}_dt_0p00390625.npz"
ONE = HERE / f"{HEAD}_materialarea_1_dt_0p00390625.npz"
SCALES = (-1/19, -1/361, 0.0, 1/361, 1/19, 6/19)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def main() -> None:
    names = ("previous", "current", "following")
    with np.load(ZERO) as archive:
        zero = {name: archive[name].copy() for name in names}
    with np.load(ONE) as archive:
        one = {name: archive[name].copy() for name in names}
    outputs = []
    for scale in SCALES:
        output = HERE / f"material_area_scan_{tag(scale)}.npz"
        blocks = {
            name: zero[name]+scale*(one[name]-zero[name]) for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"material_area_scale": scale, "output": output.name})
    report = {"status": "MATERIAL_AREA_SCAN_COMPOSED", "outputs": outputs}
    (HERE/"compose_material_area_scan.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
