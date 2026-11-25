"""Long tangent gate for Psi-derived common transport."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import psi_driven_wake_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"psi_driven_wake_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
