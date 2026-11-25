"""Full tangent gate for the live-beta kinetic metric candidate."""

from __future__ import annotations

from pathlib import Path

import live_clifford_tangent_gate as gate
import live_relative_grade_metric_action as action


gate.action = action
gate.CACHE = (Path(__file__).resolve().parent
              / "live_relative_grade_metric_tangent_blocks.npz")


if __name__ == "__main__":
    gate.main()
