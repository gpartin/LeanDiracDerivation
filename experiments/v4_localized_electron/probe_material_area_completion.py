"""Check whether the complete material frame preserves the native rest orbit."""

from __future__ import annotations

import json
import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native
import live_clifford_bond_action as action
import live_clifford_tangent_gate as gate
import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    current = bare["rest"]
    dt = native.DT
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    previous = gate.phase_rotation_matrix(current.size, -theta) @ current
    following = gate.phase_rotation_matrix(current.size, theta) @ current
    p = torch.tensor(previous, dtype=torch.float64)
    q = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    r = torch.tensor(following, dtype=torch.float64)
    residual = action.del_residual(p, q, r, dt).detach().numpy()
    report = {
        "status": "MATERIAL_AREA_REST_ORBIT_PROBE",
        "material_area_scale": action.MATERIAL_AREA_SCALE,
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_linf": float(np.max(np.abs(residual))),
        "relative_l2": float(np.linalg.norm(residual)
                             / max(np.linalg.norm(current), 1e-30)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
