"""Canonical screen of three positive reciprocal rotor-wake kinetic metrics.

These terms are research action bases, not admitted equations.  They test the
specific hypothesis that a rotating W=chi wedge Psi carrier can dynamically
prepare its C19 receivers through native interval kinetics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad, jvp, vmap

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_noncoplanar_transport_rank as transport
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = large.SIDE


def fields_and_rates(left: torch.Tensor, right: torch.Tensor):
    lp, lc = box.unpack(left, SIDE)
    rp, rc = box.unpack(right, SIDE)
    psi, chi = (lp+rp)/2, (lc+rc)/2
    pdot, cdot = (rp-lp)/small.DT, (rc-lc)/small.DT
    w = torch.linalg.cross(chi, psi, dim=-1)
    wdot = (torch.linalg.cross(cdot, psi, dim=-1)
            + torch.linalg.cross(chi, pdot, dim=-1))
    return psi, chi, pdot, cdot, w, wdot


def extra_interval(left: torch.Tensor, right: torch.Tensor,
                   mode: str) -> torch.Tensor:
    _psi, chi, _pdot, cdot, w, wdot = fields_and_rates(left, right)
    total = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        wj = torch.roll(w, shifts=offset, dims=(0, 1, 2))
        wdj = torch.roll(wdot, shifts=offset, dims=(0, 1, 2))
        if mode == "area_gradient":
            rate = wdj-wdot
            denominator = small.CHI0**2
        else:
            chij = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
            cdj = torch.roll(cdot, shifts=offset, dims=(0, 1, 2))
            zij = torch.sum(torch.conj(chi)*wj, dim=-1)
            zji = torch.sum(torch.conj(chij)*w, dim=-1)
            dzij = torch.sum(torch.conj(cdot)*wj
                              + torch.conj(chi)*wdj, dim=-1)
            dzji = torch.sum(torch.conj(cdj)*w
                              + torch.conj(chij)*wdot, dim=-1)
            _ = zij, zji  # midpoint values document the differentiated bonds
            rate = dzij+dzji if mode == "transfer_even" else dzij-dzji
            denominator = small.CHI0**4
        total += weight*torch.sum(torch.real(torch.conj(rate)*rate))
    return small.DT*total/(4*denominator)


def run(mode: str, strength: float, state: np.ndarray, fraction: float,
        phase: complex) -> dict:
    def interval(left: torch.Tensor, right: torch.Tensor):
        return (box.interval(left, right, SIDE)
                + strength*extra_interval(left, right, mode))

    def three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor):
        return interval(previous, current)+interval(current, following)

    residual = grad(three, argnums=1)
    momentum = grad(interval, argnums=1)
    current = torch.tensor(state, dtype=torch.float64)
    previous = box.phase_rotate(current, np.conjugate(phase), SIDE).detach()
    following = box.phase_rotate(current, phase, SIDE).detach()
    rest_delta = residual(previous, current, following).detach().numpy()

    canonical_residual = grad(
        lambda p, c, f: box.interval(p, c, SIDE)+box.interval(c, f, SIDE),
        argnums=1)(previous, current, following).detach().numpy()
    added = rest_delta-canonical_residual

    previous_block = torch.autograd.functional.jacobian(
        lambda value: residual(value, current, following), previous,
        vectorize=True).detach().numpy()
    singular_block = np.linalg.svd(previous_block, compute_uv=False)
    factor = lu_factor(previous_block)

    def density(left_raw: torch.Tensor, right_raw: torch.Tensor, mode_q: str):
        p = momentum(left_raw, right_raw)
        psi, chi = box.unpack(right_raw, SIDE)
        zp, zc = torch.zeros_like(psi), torch.zeros_like(chi)
        generator = torch.stack((
            torch.real(1j*psi if mode_q == "Psi" else zp),
            torch.imag(1j*psi if mode_q == "Psi" else zp),
            torch.real(1j*chi if mode_q == "chi" else zc),
            torch.imag(1j*chi if mode_q == "chi" else zc)),
            dim=-2).reshape(-1)
        return (p*generator).reshape(SIDE, SIDE, SIDE, 4, 3).sum(
            dim=(-1, -2))

    def density_jvp(left_raw, right_raw, dleft, dright, charge):
        value, derivative = jvp(
            lambda a, b: density(a, b, charge),
            (left_raw, right_raw), (dleft, dright))
        return value.detach().numpy(), derivative.detach().numpy()

    def response(launch: np.ndarray, target: np.ndarray):
        launch_t = torch.tensor(launch, dtype=torch.float64)
        target_t = torch.tensor(target, dtype=torch.float64)
        zero = torch.zeros_like(launch_t)
        cp, dcp = density_jvp(previous, current, launch_t, zero, "Psi")
        fp, dfp = density_jvp(current, following, zero, target_t, "Psi")
        _cc, dcc = density_jvp(previous, current, launch_t, zero, "chi")
        _fc, dfc = density_jvp(current, following, zero, target_t, "chi")
        charges = np.asarray(((np.sum(dcp)+np.sum(dfp))/2,
                              (np.sum(dcc)+np.sum(dfc))/2))
        center = (transport.center_derivative(fp, dfp)
                  - transport.center_derivative(cp, dcp))
        return np.r_[charges, center]

    axis_rows = []
    for axis in range(3):
        labels, targets = transport.targets(state, axis, phase)
        target_t = torch.tensor(targets.T, dtype=torch.float64)
        images = vmap(lambda vector: jvp(
            lambda value: residual(previous, current, value),
            (following,), (vector,))[1], chunk_size=4)(target_t)
        launches = lu_solve(factor, -images.detach().numpy().T)
        responses = np.column_stack([
            response(launches[:, column], targets[:, column])
            for column in range(targets.shape[1])])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        singular = np.linalg.svd(responses[2:]@null, compute_uv=False)
        threshold = 1e-8*max(singular[0], 1e-30)
        axis_rows.append({
            "axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "singular_values": [float(value) for value in singular],
            "relative_third_singular_value": float(
                singular[2]/max(singular[0], 1e-300)),
        })
    return {
        "fraction_of_rest_norm": fraction,
        "added_rest_DEL_rms": float(np.linalg.norm(added)
                                    / np.sqrt(added.size)),
        "candidate_rest_DEL_rms": float(np.linalg.norm(rest_delta)
                                        / np.sqrt(rest_delta.size)),
        "previous_block_condition_number": float(
            singular_block[0]/max(singular_block[-1], 1e-300)),
        "axis_rows": axis_rows,
        "axis_ranks": [row["fixed_charge_center_rank"] for row in axis_rows],
        "minimum_relative_third_singular_value": float(min(
            row["relative_third_singular_value"] for row in axis_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(
        "transfer_even", "transfer_odd", "area_gradient"), required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    selected = [(0.0, archive["rest"]),
                (float(archive["fractions"][-1]), archive["states"][-1])]
    rows = [run(args.mode, args.strength, state, fraction, phase)
            for fraction, state in selected]
    rank_pass = any(row["axis_ranks"] == [3, 3, 3]
                    and row["minimum_relative_third_singular_value"] > 1e-6
                    for row in rows)
    result = {
        "status": "V4_CANONICAL_TRANSFER_WAKE_KINETIC_GATE",
        "mode": args.mode,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "screen_strength": args.strength,
        "coefficient_provenance": (
            "unit dimension-normalized basis screen; no coefficient selected"),
        "rows": rows,
        "verdict": (
            "TRANSFER_WAKE_KINETIC_SUPPLIES_TRANSPORT_CHANNEL"
            if rank_pass else
            "TRANSFER_WAKE_KINETIC_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "Positive reciprocal kinetic-basis response only. A rank pass "
            "would still require an exact re-solved rest orbit, symplectic "
            "well-posedness, coefficient provenance, stability, finite "
            "translation, and same-state Dirac current."),
    }
    suffix = f"{args.strength:g}".replace("-", "m").replace(".", "p")
    output = HERE / (
        f"canonical_transfer_wake_kinetic_{args.mode}_{suffix}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "mode": args.mode, "verdict": result["verdict"],
        "rows": [{
            "fraction": row["fraction_of_rest_norm"],
            "added_rest_DEL_rms": row["added_rest_DEL_rms"],
            "condition": row["previous_block_condition_number"],
            "ranks": row["axis_ranks"],
            "minimum_relative_third": row[
                "minimum_relative_third_singular_value"]}
            for row in rows],
        "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
