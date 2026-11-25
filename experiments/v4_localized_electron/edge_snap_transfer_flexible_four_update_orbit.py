"""Four-update relative orbit with native period and phases solved jointly.

The previous best movement solve fixed dt and both endpoint phases.  This
version keeps the same action and carrier but solves those orbit data together
with the three intermediate full Psi--chi lattice layers, while retaining both
native phase-charge constraints.  No action coefficient is fitted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import density_mold_connection_probe as mold
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import edge_snap_transfer_four_update_orbit as transfer


HERE = Path(__file__).resolve().parent
UPDATES = 4
DT_MIN = 0.025
DT_MAX = 0.25
DT_INITIAL = 0.1
CHARGE_WEIGHT = 0.01
CHECKPOINT = "edge_snap_transfer_charge_four_update_orbit.npz"


def dt_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return DT_MIN + (DT_MAX - DT_MIN) * torch.sigmoid(raw)


def raw_from_dt(value: float) -> float:
    fraction = (value - DT_MIN) / (DT_MAX - DT_MIN)
    return float(np.log(fraction / (1 - fraction)))


def interval_action(left, right, dt):
    return transfer.interval_action(left, right, dt, mold.ALPHA)


def del_at(previous, current, following, dt):
    action = (interval_action(previous, current, dt)
              + interval_action(current, following, dt))
    return torch.autograd.grad(action, current, create_graph=True)[0]


def interval_charge(left, right, dt, field: str,
                    create_graph: bool = True):
    if not right.requires_grad:
        right = right.detach().clone().requires_grad_(True)
    action = interval_action(left, right, dt)
    momentum = torch.autograd.grad(
        action, right, create_graph=create_graph)[0]
    return torch.dot(momentum, four.phase_generator(right, field))


def charge_rows(states, dt, create_graph=True):
    return {
        field: [interval_charge(states[index], states[index + 1], dt, field,
                                create_graph=create_graph)
                for index in range(UPDATES)]
        for field in ("psi", "chi")
    }


def main(iterations: int = 600) -> None:
    rank.lock_inertia()
    transfer.GEOMETRIC_FACTOR = 19.0
    transfer.FILTER_FACTOR = 0.0
    transfer.COLLECTIVE_FACTOR = 0.0
    transfer.RADIAL_FACTOR = 0.0
    transfer.NORMAL_FACTOR = -1.0
    transfer.BOND_FRAME_FACTOR = 0.0
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(four.orbit.SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    phase_psi = torch.tensor(
        four.PHASE_PSI, dtype=torch.float64, requires_grad=True)
    phase_chi = torch.tensor(
        four.PHASE_CHI, dtype=torch.float64, requires_grad=True)
    raw_dt = torch.tensor(
        raw_from_dt(DT_INITIAL), dtype=torch.float64, requires_grad=True)
    checkpoint = np.load(HERE / CHECKPOINT)
    middle = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
              for value in checkpoint["middle"]]
    parameters = middle + [raw_dt, phase_psi, phase_chi]
    optimizer = torch.optim.LBFGS(
        parameters, lr=0.35, max_iter=iterations, max_eval=iterations * 2,
        tolerance_grad=1e-11, tolerance_change=1e-15,
        history_size=30, line_search_fn="strong_wolfe")
    scale = rank.B * rank.R / DT_INITIAL

    def states():
        qn = four.orbit.translate(
            q0, four.orbit.SIGN, phase_psi, phase_chi)
        return [q0] + middle + [qn]

    with torch.enable_grad():
        initial_states = states()
        initial_charges = charge_rows(
            initial_states, dt_from_raw(raw_dt), create_graph=False)
    charge_scales = {
        field: max(max(abs(float(value.detach())) for value in values), 1.0)
        for field, values in initial_charges.items()
    }
    trace = []

    def residual_rows():
        dt = dt_from_raw(raw_dt)
        all_states = states()
        qm1 = four.orbit.translate(
            middle[-1], -four.orbit.SIGN, -phase_psi, -phase_chi)
        q0_work = q0.detach().clone().requires_grad_(True)
        rows = [del_at(qm1, q0_work, middle[0], dt)]
        for index in range(1, UPDATES):
            rows.append(del_at(all_states[index - 1], middle[index - 1],
                               all_states[index + 1], dt))
        return rows, all_states, dt

    def closure():
        optimizer.zero_grad()
        rows, all_states, dt = residual_rows()
        del_loss = sum(torch.mean((row / scale)**2) for row in rows)
        charges = charge_rows(all_states, dt)
        charge_loss = torch.zeros((), dtype=torch.float64)
        for field, values in charges.items():
            reference = values[0]
            charge_loss = charge_loss + sum(
                ((value - reference) / charge_scales[field])**2
                for value in values[1:])
        loss = del_loss + CHARGE_WEIGHT * charge_loss
        loss.backward()
        trace.append({
            "combined": float(torch.sqrt(loss.detach())),
            "del": float(torch.sqrt(del_loss.detach())),
            "dt": float(dt.detach()),
            "phase_psi": float(phase_psi.detach()),
            "phase_chi": float(phase_chi.detach()),
        })
        return loss

    initial = float(torch.sqrt(closure().detach()))
    optimizer.step(closure)
    rows, final_states, dt = residual_rows()
    final_del = float(torch.sqrt(sum(
        torch.mean((row / scale)**2) for row in rows)).detach())
    charges = charge_rows(final_states, dt, create_graph=False)
    charge_defects = {
        field: float(max(
            abs(float(value.detach() - values[0].detach()))
            / charge_scales[field] for value in values[1:]))
        for field, values in charges.items()
    }
    final = float(np.sqrt(
        final_del**2 + sum(value**2 for value in charge_defects.values())))
    qn = final_states[-1]
    result = {
        "status": (
            "V4_FLEXIBLE_4_UPDATE_EDGE_ORBIT_PASS" if final < 1e-8 else
            "V4_FLEXIBLE_4_UPDATE_EDGE_ORBIT_INCONCLUSIVE"),
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_variant": "chi_transfer_softening_factor_19",
        "updates_per_translation": UPDATES,
        "initial_combined_residual": initial,
        "final_scaled_del_rms": final_del,
        "final_combined_residual": final,
        "solved_dt": float(dt.detach()),
        "solved_translation_time": float(UPDATES * dt.detach()),
        "solved_phase_psi": float(phase_psi.detach()),
        "solved_phase_chi": float(phase_chi.detach()),
        "maximum_normalized_noether_charge_defects": charge_defects,
        "optimizer_evaluations": len(trace),
        "trace_tail": trace[-10:],
        "endpoint_constraint_defect": float(torch.linalg.vector_norm(
            qn - four.orbit.translate(
                q0, four.orbit.SIGN, phase_psi, phase_chi)).detach()),
        "verdict": (
            "EXACT_RELATIVE_PERIODIC_EDGE_ORBIT_FOUND" if final < 1e-8 else
            "FREE_PERIOD_PHASE_AND_STATE_SEARCH_DOES_NOT_YET_CLOSE"),
        "claim_boundary": (
            "One freely optimized four-update relative orbit. Failure is "
            "inconclusive; solved orbit data are not action coefficients."),
    }
    np.savez_compressed(
        HERE / "edge_snap_transfer_flexible_four_update_orbit.npz",
        q0=q0.detach().numpy(),
        middle=np.stack([value.detach().numpy() for value in middle]),
        qn=qn.detach().numpy(),
        residuals=np.stack([row.detach().numpy() for row in rows]))
    (HERE / "edge_snap_transfer_flexible_four_update_orbit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
