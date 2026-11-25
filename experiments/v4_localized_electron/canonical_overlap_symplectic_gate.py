"""Canonical V40 screen of native Psi--chi overlap one-forms.

The position-action transport rank is kinematically fixed.  These are the two
lowest reciprocal, cubic-symmetric velocity-linear interactions in the local
parallel-overlap channel S=chi^dagger Psi:

  local: Im(conj(S_i) Sdot_i)
  bond:  sum_C19 w Im(conj(S_i) Sdot_j + conj(S_j) Sdot_i)/2

Both vanish with first variation on the coplanar carrier and the exact
side-five chi-orientation family.  Unit normalization is a response-basis
screen, not a selected action coefficient.
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


def overlap_and_rate(left: torch.Tensor, right: torch.Tensor):
    lp, lc = box.unpack(left, SIDE)
    rp, rc = box.unpack(right, SIDE)
    psi = (lp+rp)/2
    chi = (lc+rc)/2
    psi_rate = (rp-lp)/small.DT
    chi_rate = (rc-lc)/small.DT
    overlap = torch.sum(torch.conj(chi)*psi, dim=-1)
    rate = torch.sum(torch.conj(chi_rate)*psi
                     + torch.conj(chi)*psi_rate, dim=-1)
    return overlap, rate


def local_one_form(left: torch.Tensor, right: torch.Tensor):
    overlap, rate = overlap_and_rate(left, right)
    return torch.sum(torch.imag(torch.conj(overlap)*rate))/small.CHI0**2


def bond_one_form(left: torch.Tensor, right: torch.Tensor):
    overlap, rate = overlap_and_rate(left, right)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        neighbor_overlap = torch.roll(overlap, shifts=offset,
                                      dims=(0, 1, 2))
        neighbor_rate = torch.roll(rate, shifts=offset, dims=(0, 1, 2))
        result += weight*torch.sum(torch.imag(
            torch.conj(overlap)*neighbor_rate
            + torch.conj(neighbor_overlap)*rate))
    return result/(2*small.CHI0**2)


def run(mode: str, state: np.ndarray, fraction: float,
        phase: complex) -> dict:
    one_form = local_one_form if mode == "local" else bond_one_form

    def interval(left: torch.Tensor, right: torch.Tensor):
        return box.interval(left, right, SIDE) \
            + small.DT*one_form(left, right)

    def three(previous: torch.Tensor, current: torch.Tensor,
              following: torch.Tensor):
        return interval(previous, current)+interval(current, following)

    del_residual = grad(three, argnums=1)
    momentum = grad(interval, argnums=1)
    current = torch.tensor(state, dtype=torch.float64)
    previous = box.phase_rotate(
        current, np.conjugate(phase), SIDE).detach()
    following = box.phase_rotate(current, phase, SIDE).detach()
    layers = (previous, current, following)
    rest_value = float(one_form(previous, current).detach())
    left = previous.detach().requires_grad_(True)
    right = current.detach().requires_grad_(True)
    first_variation = torch.autograd.grad(one_form(left, right), (left, right))
    first_variation_norm = float(torch.sqrt(sum(
        torch.sum(value*value) for value in first_variation)).detach())
    rest_del = del_residual(*layers).detach().numpy()

    previous_block = torch.autograd.functional.jacobian(
        lambda value: del_residual(value, current, following), previous,
        vectorize=True).detach().numpy()
    factor = lu_factor(previous_block)

    def density_tensor(left_raw: torch.Tensor, right_raw: torch.Tensor,
                       charge_mode: str):
        p = momentum(left_raw, right_raw)
        psi, chi = box.unpack(right_raw, SIDE)
        zp, zc = torch.zeros_like(psi), torch.zeros_like(chi)
        generator = torch.stack((
            torch.real(1j*psi if charge_mode == "Psi" else zp),
            torch.imag(1j*psi if charge_mode == "Psi" else zp),
            torch.real(1j*chi if charge_mode == "chi" else zc),
            torch.imag(1j*chi if charge_mode == "chi" else zc)),
            dim=-2).reshape(-1)
        return (p*generator).reshape(SIDE, SIDE, SIDE, 4, 3).sum(
            dim=(-1, -2))

    def density_derivative(left_raw, right_raw, dleft, dright, charge_mode):
        value, derivative = jvp(
            lambda a, b: density_tensor(a, b, charge_mode),
            (left_raw, right_raw), (dleft, dright))
        return value.detach().numpy(), derivative.detach().numpy()

    def response(launch: np.ndarray, target: np.ndarray):
        launch_t = torch.tensor(launch, dtype=torch.float64)
        target_t = torch.tensor(target, dtype=torch.float64)
        zero = torch.zeros_like(launch_t)
        cp, dcp = density_derivative(
            previous, current, launch_t, zero, "Psi")
        fp, dfp = density_derivative(
            current, following, zero, target_t, "Psi")
        _cc, dcc = density_derivative(
            previous, current, launch_t, zero, "chi")
        _fc, dfc = density_derivative(
            current, following, zero, target_t, "chi")
        charges = np.asarray(((np.sum(dcp)+np.sum(dfp))/2,
                              (np.sum(dcc)+np.sum(dfc))/2))
        center = (transport.center_derivative(fp, dfp)
                  - transport.center_derivative(cp, dcp))
        return np.r_[charges, center]

    rows = []
    for axis in range(3):
        labels, targets = transport.targets(state, axis, phase)
        target_tensor = torch.tensor(targets.T, dtype=torch.float64)
        images = vmap(lambda vector: jvp(
            lambda value: del_residual(previous, current, value),
            (following,), (vector,))[1], chunk_size=4)(
                target_tensor).detach().numpy().T
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(launches[:, index], targets[:, index])
            for index in range(targets.shape[1])])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed = responses[2:]@null
        singular = np.linalg.svd(fixed, compute_uv=False)
        threshold = 1e-8*max(singular[0], 1e-30)
        rows.append({
            "axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "singular_values": [float(value) for value in singular],
            "relative_third_singular_value": float(
                singular[2]/max(singular[0], 1e-300)),
        })
    return {
        "fraction_of_rest_norm": fraction,
        "rest_one_form_value": rest_value,
        "rest_one_form_first_variation_norm": first_variation_norm,
        "rest_DEL_rms": float(np.linalg.norm(rest_del)
                              / np.sqrt(rest_del.size)),
        "axis_rows": rows,
        "axis_ranks": [row["fixed_charge_center_rank"] for row in rows],
        "minimum_relative_third_singular_value": float(min(
            row["relative_third_singular_value"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "bond"), required=True)
    args = parser.parse_args()
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    rest = archive["rest"]
    state = archive["states"][-1]
    fraction = float(archive["fractions"][-1])
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    rows = [run(args.mode, rest, 0.0, phase),
            run(args.mode, state, fraction, phase)]
    passing = [row for row in rows
               if row["rest_DEL_rms"] < 1e-9
               and row["axis_ranks"] == [3, 3, 3]
               and row["minimum_relative_third_singular_value"] > 1e-6]
    result = {
        "status": "V4_CANONICAL_OVERLAP_SYMPLECTIC_GATE",
        "mode": args.mode,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "screen_strength": 1.0,
        "coefficient_provenance": (
            "unit geometric normalization only; no coefficient selected"),
        "rows": rows,
        "verdict": (
            "CANONICAL_OVERLAP_SYMPLECTIC_SUPPLIES_TRANSPORT_CHANNEL"
            if passing else
            "CANONICAL_OVERLAP_SYMPLECTIC_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "Unit-strength linear response screen on canonical rest states. "
            "No coefficient admission, nonlinear motion, stability, or "
            "Dirac-current claim."),
    }
    output = HERE / f"canonical_overlap_symplectic_{args.mode}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "verdict": result["verdict"],
        "rows": [{
            "fraction": row["fraction_of_rest_norm"],
            "rest_DEL_rms": row["rest_DEL_rms"],
            "first_variation": row["rest_one_form_first_variation_norm"],
            "ranks": row["axis_ranks"],
            "minimum_relative_third": row[
                "minimum_relative_third_singular_value"]}
            for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
