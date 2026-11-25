"""Long tangent gate for the fully reciprocal live-link action candidate.

This deliberately reuses the established phase-space integrator and packet
diagnostics.  Only the action and tangent cache are replaced.  The action
derives its link and three-prong transporter from the instantaneous native
Psi/chi fields; no Clifford matrix is supplied to the force.
"""

from __future__ import annotations

import os
from pathlib import Path


# Set the admissible, fully reciprocal research branch before importing the
# action module.  The frozen-link controls are diagnostic only and are never
# used by this gate.
os.environ.setdefault("LFM_LIVE_CLIFFORD_LINK", "psi_frame_phase")
os.environ.setdefault("LFM_LIVE_CLIFFORD_SCALE", "1")
os.environ.setdefault("LFM_LIVE_CLIFFORD_CHI_PAYLOAD", "1")

import live_clifford_bond_action as action  # noqa: E402
import live_clifford_tangent_gate as gate  # noqa: E402


gate.action = action
scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
gate.CACHE = (Path(__file__).resolve().parent
              / f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}.npz")


if __name__ == "__main__":
    gate.main()
