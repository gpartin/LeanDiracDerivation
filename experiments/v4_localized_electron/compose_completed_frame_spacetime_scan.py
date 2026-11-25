"""Re-scan geometric spacetime strengths after frame completion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HEAD = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_")
TAIL_OLD = "_symplectic_0_dt_0p00390625.npz"
MATCHED = HERE / (
    f"{HEAD}0p00554016620499_symplectic_0_overlap_0_overlapface_0_"
    "overlapmatched_1_overlappotential_1_dt_0p00390625.npz")
BASE_TWO = HERE / f"{HEAD}0p00554016620499{TAIL_OLD}"
BASE_ZERO = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0_symplectic_0.npz")
SCALES = (0.0, 1/361, 2/361, 3/361, 0.012, 0.014, 0.016,
          6/361, 0.018, 0.020, 1/38, 1/19)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def load(path: Path, names: tuple[str, ...]):
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in names}


def main() -> None:
    names = ("previous", "current", "following")
    matched = load(MATCHED, names)
    base_two = load(BASE_TWO, names)
    base_zero = load(BASE_ZERO, names)
    matched_delta = {
        name: matched[name]-base_two[name] for name in names
    }
    spacetime_per_unit = {
        name: (base_two[name]-base_zero[name])/(2/361)
        for name in names
    }
    outputs = []
    for scale in SCALES:
        base = {
            name: base_zero[name]+scale*spacetime_per_unit[name]
            for name in names
        }
        output = HERE / f"completed_spacetime_scan_{tag(scale)}.npz"
        blocks = {
            name: base[name]+matched_delta[name] for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"spacetime_scale": scale, "output": output.name})
    report = {
        "status": "V4_COMPLETED_FRAME_SPACETIME_SCAN_COMPOSED",
        "identity": "candidate(s) = base(s) + fixed matched-frame delta",
        "outputs": outputs,
    }
    (HERE / "compose_completed_frame_spacetime_scan.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
