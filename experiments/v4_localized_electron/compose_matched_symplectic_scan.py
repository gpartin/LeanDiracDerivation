"""Compose both native face-loop chirality signs on the repaired frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FRAME_PREFIX = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499")
BASE = HERE / (
    f"{FRAME_PREFIX}_symplectic_0_overlap_0_overlapface_0_"
    "overlapmatched_1_overlappotential_1_dt_0p00390625.npz")
PSI_PREFIX = (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_psi_area_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0")
SYM_UNIT = HERE / f"{PSI_PREFIX}_symplectic_0p0526315789474.npz"
SYM_ZERO = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_psi_area_oscillator_0p315789473684_"
    "plaquette_0.npz")
SCALES = (-1/19, -1/361, 1/361, 1/19)


def tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def load(path: Path, names: tuple[str, ...]):
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in names}


def main() -> None:
    names = ("previous", "current", "following")
    base = load(BASE, names)
    sym_unit = load(SYM_UNIT, names)
    sym_zero = load(SYM_ZERO, names)
    delta_per_unit = {
        name: (sym_unit[name]-sym_zero[name])/(1/19)
        for name in names
    }
    outputs = []
    for scale in SCALES:
        output = HERE / (
            f"{FRAME_PREFIX}_symplectic_{tag(scale)}_overlap_0_"
            "overlapface_0_overlapmatched_1_overlappotential_1_"
            "dt_0p00390625.npz")
        blocks = {
            name: base[name]+scale*delta_per_unit[name]
            for name in names
        }
        np.savez_compressed(output, **blocks)
        outputs.append({"symplectic_scale": scale, "output": output.name})
    report = {
        "status": "V4_MATCHED_SYMPLECTIC_SCAN_COMPOSED",
        "identity": "candidate = repaired_frame + s * symplectic_unit",
        "outputs": outputs,
    }
    (HERE / "compose_matched_symplectic_scan.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
