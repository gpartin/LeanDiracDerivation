"""Assemble exact tangent blocks for the movement action.

PyTorch's batched ``jacrev`` has no batching rule for one conjugate-view
operation used by the cube one-form.  The one-form's audited M/G/K blocks are
already available.  This module algebraically inverts their definitions to
the previous/current/following DEL Jacobians and adds them to the incumbent
blocks.  No finite difference or fitted quantity is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as prior_action
import cube_incidence_symplectic_completion_gate as completion


HERE = Path(__file__).resolve().parent
INCUMBENT = HERE / "wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz"
OUTPUT = HERE / "cube_symplectic_movement_tangent_blocks.npz"
REPORT = HERE / "cube_symplectic_movement_blocks.json"
SCALE = float(os.environ.get("LFM_CUBE_SYMPLECTIC_SCALE", "1"))
VARIANT = os.environ.get("LFM_CUBE_SYMPLECTIC_VARIANT", "overlap")


def main() -> None:
    incumbent = np.load(INCUMBENT)
    if VARIANT in ("gradient", "complete"):
        raw_delta = np.load(completion.CACHE)
        delta = {name: raw_delta[name]
                 for name in ("mass", "gyro", "stiffness")}
        if VARIANT == "complete":
            complete = np.load(
                HERE / "cube_incidence_complete_frame_matrices.npz")
            delta = {
                name: -0.5*delta[name]+complete[f"overlap_{name}"]
                for name in ("mass", "gyro", "stiffness")
            }
    else:
        raw_delta = np.load(HERE / "cube_incidence_complete_frame_matrices.npz")
        delta = {name: raw_delta[f"{VARIANT}_{name}"]
                 for name in ("mass", "gyro", "stiffness")}
    rest, _ = prior_action.expected_blocks()
    q0 = rest["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(q0.size, theta)

    mass = delta["mass"]
    gyro = delta["gyro"]
    stiffness = delta["stiffness"]
    plus_rotated = -mass/dt-gyro/2
    minus_rotated = -mass/dt+gyro/2
    zero = -dt*stiffness+2*mass/dt
    cube_previous = minus_rotated@rotation
    cube_following = plus_rotated@rotation.T

    blocks = {
        "previous": incumbent["previous"]+SCALE*cube_previous,
        "current": incumbent["current"]+SCALE*zero,
        "following": incumbent["following"]+SCALE*cube_following,
    }
    np.savez_compressed(OUTPUT, **blocks)
    np.savez_compressed(
        HERE/f"cube_symplectic_movement_{VARIANT}_tangent_blocks.npz",
        **blocks)

    reconstructed = {
        "mass": -0.5*dt*(cube_following@rotation
                          + cube_previous@rotation.T),
        "gyro": -(cube_following@rotation
                   - cube_previous@rotation.T),
        "stiffness": -(cube_following@rotation+zero
                       + cube_previous@rotation.T)/dt,
    }
    errors = {
        name: float(np.linalg.norm(reconstructed[name]-delta[name])
                    / max(np.linalg.norm(delta[name]), 1e-300))
        for name in reconstructed
    }
    result = {
        "status": "V4_CUBE_SYMPLECTIC_MOVEMENT_BLOCKS",
        "incumbent": INCUMBENT.name,
        "cube_variant": VARIANT,
        "cube_scale": SCALE,
        "assembly": "exact linear action-block addition",
        "relative_reconstruction_errors": errors,
        "maximum_relative_reconstruction_error": max(errors.values()),
        "verdict": ("EXACT_BLOCK_ASSEMBLY_PASS"
                    if max(errors.values()) < 1e-12
                    else "BLOCK_ASSEMBLY_FAIL"),
    }
    REPORT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
