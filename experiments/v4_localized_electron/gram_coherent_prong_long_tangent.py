"""Long tangent gate for full live-metric prong coherence."""

from __future__ import annotations

from pathlib import Path

import gram_coherent_prong_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"gram_coherent_prong_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
