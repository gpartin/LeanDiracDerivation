"""Compose the exact tangent cache for face oscillator plus C19 triangle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OSCILLATOR = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_psi_area_oscillator_0p315789473684.npz")
TRIANGLE = HERE / "triangle_movement_blocks.npz"
TRIANGLE_BASE = HERE / (
    "wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
OUTPUT = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy_"
    "0p315789473684.npz")


def main() -> None:
    oscillator = np.load(OSCILLATOR)
    triangle = np.load(TRIANGLE)
    triangle_base = np.load(TRIANGLE_BASE)
    names = ("previous", "current", "following")
    blocks = {
        name: oscillator[name]+triangle[name]-triangle_base[name]
        for name in names
    }
    np.savez_compressed(OUTPUT, **blocks)
    result = {
        "status": "V4_LIVE_OSCILLATOR_TRIANGLE_TANGENT_CACHE_COMPOSED",
        "linearity_identity": (
            "combined = oscillator + triangle - triangle_base"),
        "output": OUTPUT.name,
    }
    (HERE/"live_oscillator_triangle_tangent_cache.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
