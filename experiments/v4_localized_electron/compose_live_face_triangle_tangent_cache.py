"""Compose the exact tangent blocks of two additive interval actions.

The live six-face metric cache and the triangle cache were differentiated
independently at the same retained rest orbit.  Since Euler--Lagrange
variation and linearization are linear in the action, their combined blocks
are the face blocks plus the already-scaled triangle delta.  This avoids a
large automatic-differentiation graph without changing either action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FACE = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_wedge_laplacian_smooth_0p0526315789474.npz")
TRIANGLE = HERE / "triangle_movement_blocks.npz"
TRIANGLE_BASE = HERE / (
    "wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
OUTPUT = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_triangle_holonomy_0p0526315789474.npz")


def main() -> None:
    face = np.load(FACE)
    triangle = np.load(TRIANGLE)
    triangle_base = np.load(TRIANGLE_BASE)
    names = ("previous", "current", "following")
    blocks = {
        name: face[name] + triangle[name] - triangle_base[name]
        for name in names
    }
    np.savez_compressed(OUTPUT, **blocks)
    report = {
        "status": "V4_LIVE_FACE_TRIANGLE_TANGENT_CACHE_COMPOSED",
        "linearity_identity": "combined = face + triangle - triangle_base",
        "triangle_delta_norms": {
            name: float(np.linalg.norm(triangle[name]-triangle_base[name]))
            for name in names
        },
        "output": OUTPUT.name,
    }
    (HERE / "live_face_triangle_tangent_cache.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
