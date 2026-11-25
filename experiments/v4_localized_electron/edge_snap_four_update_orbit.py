"""Four-update relative-periodic one-edge orbit search.

The two-update solve saturated its allowed physical interval.  This test keeps
the same one-edge endpoint and native action but gives the fields three full
intermediate lattice layers in which to carry bond current and reshape chi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank
import edge_snap_relative_orbit_search as orbit


HERE = Path(__file__).resolve().parent
UPDATES = 4
DT = 0.1
PHASE_PSI = 3.3287798462704368
PHASE_CHI = -6.889772046440938e-05
CHARGE_WEIGHT = 1e-4


def del_at(previous: torch.Tensor, current: torch.Tensor,
           following: torch.Tensor):
    action = (orbit.interval_action(previous, current, DT, mold.ALPHA)
              + orbit.interval_action(current, following, DT, mold.ALPHA))
    return torch.autograd.grad(action, current, create_graph=True)[0]


def initial_path(q0: torch.Tensor, qn: torch.Tensor):
    states = []
    for index in range(1, UPDATES):
        fraction = index / UPDATES
        angle = np.pi * fraction / 2
        state = np.cos(angle) * q0 + np.sin(angle) * qn
        states.append(state.detach().clone().requires_grad_(True))
    return states


def phase_generator(raw: torch.Tensor, field: str) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, orbit.SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator_psi = 1j * psi if field == "psi" else zero_psi
    generator_chi = 1j * chi if field == "chi" else zero_chi
    return torch.stack((generator_psi.real, generator_psi.imag,
                        generator_chi.real, generator_chi.imag),
                       dim=-2).reshape(-1)


def interval_charge(left: torch.Tensor, right: torch.Tensor, field: str,
                    create_graph: bool = True) -> torch.Tensor:
    if not right.requires_grad:
        right = right.detach().clone().requires_grad_(True)
    action = orbit.interval_action(left, right, DT, mold.ALPHA)
    momentum = torch.autograd.grad(
        action, right, create_graph=create_graph)[0]
    return torch.dot(momentum, phase_generator(right, field))


def charge_rows(states, create_graph: bool = True):
    return {
        field: [interval_charge(states[index], states[index + 1], field,
                                create_graph=create_graph)
                for index in range(UPDATES)]
        for field in ("psi", "chi")
    }


def main(iterations: int = 300,
         output_stem: str = "edge_snap_four_update_orbit",
         action_variant: str = "local_density_mold",
         initial_middle: np.ndarray | None = None):
    rank.lock_inertia()
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(orbit.SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    qn = orbit.translate(q0, orbit.SIGN, PHASE_PSI, PHASE_CHI).detach()
    if initial_middle is None:
        middle = initial_path(q0, qn)
    else:
        middle = [torch.tensor(value, dtype=torch.float64,
                               requires_grad=True)
                  for value in initial_middle]
    initial_states = [q0] + middle + [qn]
    initial_charges = charge_rows(initial_states, create_graph=False)
    charge_scales = {
        field: max(max(abs(float(value.detach())) for value in values), 1.0)
        for field, values in initial_charges.items()
    }
    optimizer = torch.optim.LBFGS(
        middle, lr=0.35, max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-11, tolerance_change=1e-15,
        history_size=30, line_search_fn="strong_wolfe")
    scale = rank.B * rank.R / DT
    trace = []

    def residuals():
        qm1 = orbit.translate(
            middle[-1], -orbit.SIGN, -PHASE_PSI, -PHASE_CHI)
        all_states = [q0] + middle + [qn]
        rows = [del_at(qm1, q0.detach().clone().requires_grad_(True),
                       middle[0])]
        for index in range(1, UPDATES):
            rows.append(del_at(all_states[index - 1], middle[index - 1],
                               all_states[index + 1]))
        return rows

    def closure():
        optimizer.zero_grad()
        rows = residuals()
        loss = sum(torch.mean((row / scale)**2) for row in rows)
        states = [q0] + middle + [qn]
        charges = charge_rows(states)
        for field, values in charges.items():
            reference = values[0]
            loss = loss + CHARGE_WEIGHT * sum(
                ((value - reference) / charge_scales[field])**2
                for value in values[1:])
        loss.backward()
        trace.append(float(torch.sqrt(loss.detach())))
        return loss

    initial_rows = residuals()
    initial_del = float(torch.sqrt(sum(
        torch.mean((row / scale)**2) for row in initial_rows)).detach())
    initial = float(torch.sqrt(closure().detach()))
    optimizer.step(closure)
    rows = residuals()
    final_del = float(torch.sqrt(sum(
        torch.mean((row / scale)**2) for row in rows)).detach())
    final_charges = charge_rows([q0] + middle + [qn], create_graph=False)
    charge_defects = {
        field: float(max(
            abs(float(value.detach() - values[0].detach()))
            / charge_scales[field] for value in values[1:]))
        for field, values in final_charges.items()
    }
    final = float(np.sqrt(
        final_del**2 + sum(value**2 for value in charge_defects.values())))
    density_sums = []
    centers = []
    coordinates = torch.tensor(
        np.indices((orbit.SIDE, orbit.SIDE, orbit.SIDE)).reshape(3, -1).T,
        dtype=torch.float64)
    for state in middle:
        psi_state, _ = mold.unpack_tensor(state.detach(), orbit.SIDE)
        density = torch.sum(mold.squared_norm(psi_state), dim=-1)
        density_sums.append(float(torch.sum(density)))
        centers.append((torch.sum(
            coordinates * density.reshape(-1, 1), dim=0)
            / torch.sum(density)).detach().numpy().tolist())
    result = {
        "status": (f"V4_{UPDATES}_UPDATE_EDGE_ORBIT_PASS" if final < 1e-8 else
                   f"V4_{UPDATES}_UPDATE_EDGE_ORBIT_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_variant": action_variant,
        "updates_per_translation": UPDATES,
        "dt": DT,
        "translation_time": UPDATES * DT,
        "phase_psi": PHASE_PSI,
        "phase_chi": PHASE_CHI,
        "noether_charge_penalty_weight": CHARGE_WEIGHT,
        "initial_scaled_del_rms": initial_del,
        "initial_combined_del_and_charge_residual": initial,
        "final_scaled_del_rms": final_del,
        "final_combined_del_and_charge_residual": final,
        "relative_del_residual_reduction": float(
            1 - final_del / initial_del),
        "maximum_normalized_noether_charge_defects": charge_defects,
        "optimizer_evaluations": len(trace),
        "minimum_trace_residual": min(trace),
        "intermediate_psi_density_sums": density_sums,
        "intermediate_centers_of_density": centers,
        "endpoint_constraint_defect": float(torch.linalg.vector_norm(
            qn - orbit.translate(q0, orbit.SIGN, PHASE_PSI, PHASE_CHI))),
        "verdict": (
            "EXACT_RELATIVE_PERIODIC_EDGE_ORBIT_FOUND" if final < 1e-8 else
            f"{UPDATES}_UPDATE_PATH_DOES_NOT_YET_CLOSE"),
        "claim_boundary": (
            "This is one nonlinear boundary-value search. Nonconvergence is "
            "not a no-go theorem and does not admit the research action."),
    }
    np.savez_compressed(
        HERE / f"{output_stem}.npz",
        q0=q0.detach().numpy(),
        middle=np.stack([state.detach().numpy() for state in middle]),
        qn=qn.detach().numpy(),
        residuals=np.stack([row.detach().numpy() for row in rows]))
    (HERE / f"{output_stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
