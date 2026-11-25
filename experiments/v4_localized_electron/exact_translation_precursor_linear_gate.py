"""Test whether a native Psi--chi precursor can close the exact cell shift.

The non-negotiable endpoint is the literal same-assignment translation of the
complete joint field profile by one lattice cell.  No destination is reset and
no copying force is installed.  Instead, this script keeps every occupied
carrier-core value exactly equal to the proved rest carrier and asks whether
small native fields on the surrounding sites can make the canonical centered
update satisfy

    q[n+1] = phase * shift(q[n]),
    q[n-1] = phase^-1 * shift^-1(q[n]).

This is the linear inverse gate for a freely carried precursor/halo.  A small
nonlinear residual would justify a full nonlinear profile solve.  Failure is
scoped to the canonical action linearized at the compact carrier and does not
license a runtime copy rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch

import canonical_relative_periodic_cell_orbit as orbit
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent
SIDE = 5


def translate(raw: torch.Tensor, axis: int, sign: int,
              psi_phase: float, chi_phase: float) -> torch.Tensor:
    psi, chi = orbit.unpack(raw)
    pp = torch.exp(1j * torch.as_tensor(psi_phase, dtype=torch.float64))
    cp = torch.exp(1j * torch.as_tensor(chi_phase, dtype=torch.float64))
    return orbit.pack(
        torch.roll(psi, sign, dims=axis) * pp,
        torch.roll(chi, sign, dims=axis) * cp)


def recurrence(raw: torch.Tensor, axis: int, sign: int, dt: float,
               psi_phase: float, chi_phase: float) -> torch.Tensor:
    following = translate(raw, axis, sign, psi_phase, chi_phase)
    previous = translate(raw, axis, -sign, -psi_phase, -chi_phase)
    return following - 2 * raw + previous - dt**2 * orbit.canonical_force(raw)


def site_masks(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    core = np.sum(abs(psi)**2, axis=-1) > 1e-12
    shell = np.zeros_like(core)
    core_sites = np.argwhere(core)
    for site in core_sites:
        for offset, _weight in rank.native.DIRECTIONS:
            neighbor = tuple((int(site[d]) + int(offset[d])) % SIDE
                             for d in range(3))
            if not core[neighbor]:
                shell[neighbor] = True
    return core, shell


def raw_coordinate_mask(site_mask: np.ndarray) -> np.ndarray:
    # pack order is site, (Psi.re, Psi.im, chi.re, chi.im), component.
    return np.repeat(site_mask[..., None, None], 4, axis=-2,
                     ).repeat(3, axis=-1).reshape(-1)


def residual_diagnostics(raw: torch.Tensor, core: np.ndarray,
                         shell: np.ndarray) -> dict:
    shaped = raw.detach().numpy().reshape(SIDE, SIDE, SIDE, 4, 3)
    psi_power = np.sum(shaped[..., 0:2, :]**2, axis=(-2, -1))
    chi_power = np.sum(shaped[..., 2:4, :]**2, axis=(-2, -1))
    total = psi_power + chi_power
    maximum = np.unravel_index(int(np.argmax(total)), total.shape)
    component_power = np.sum(shaped**2, axis=(0, 1, 2, 3))
    denominator = max(float(np.sum(total)), 1e-300)
    return {
        "Psi_power_fraction": float(np.sum(psi_power) / denominator),
        "chi_power_fraction": float(np.sum(chi_power) / denominator),
        "core_power_fraction": float(np.sum(total[core]) / denominator),
        "first_shell_power_fraction": float(
            np.sum(total[shell]) / denominator),
        "far_exterior_power_fraction": float(
            np.sum(total[~(core | shell)]) / denominator),
        "complex_component_power_fractions": [
            float(value / denominator) for value in component_power],
        "maximum_residual_site": [int(value) for value in maximum],
        "maximum_residual_site_power_fraction": float(
            total[maximum] / denominator),
    }


def solve_case(name: str, free_sites: np.ndarray, core: np.ndarray,
               shell: np.ndarray, q0: torch.Tensor,
               residual0: torch.Tensor, function, maxiter: int) -> dict:
    coordinate_mask = raw_coordinate_mask(free_sites)
    free_indices = np.flatnonzero(coordinate_mask)
    m = q0.numel()
    n = free_indices.size

    def lift(values: np.ndarray) -> torch.Tensor:
        full = torch.zeros_like(q0)
        full[torch.tensor(free_indices, dtype=torch.long)] = torch.tensor(
            values, dtype=torch.float64)
        return full

    def matvec(values: np.ndarray) -> np.ndarray:
        direction = lift(np.asarray(values, dtype=float))
        _value, image = torch.autograd.functional.jvp(
            function, q0, direction, create_graph=False, strict=False)
        return image.detach().numpy()

    def rmatvec(values: np.ndarray) -> np.ndarray:
        probe = torch.tensor(np.asarray(values, dtype=float),
                             dtype=torch.float64)
        variable = q0.detach().clone().requires_grad_(True)
        value = function(variable)
        gradient = torch.autograd.grad(
            torch.dot(value, probe), variable, create_graph=False)[0]
        return gradient.detach().numpy()[free_indices]

    operator = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec,
                              dtype=np.float64)
    solution = lsmr(operator, -residual0.detach().numpy(),
                    atol=1e-10, btol=1e-10, maxiter=maxiter,
                    conlim=1e10)
    correction = lift(solution[0])
    linear_residual = residual0.detach().numpy() + matvec(solution[0])
    baseline_norm = float(torch.linalg.vector_norm(residual0))

    scales = np.concatenate((np.linspace(0.0, 0.1, 11),
                             np.linspace(0.2, 1.0, 9)))
    nonlinear_rows = []
    for scale in scales:
        trial = q0 + float(scale) * correction
        residual = function(trial).detach()
        nonlinear_rows.append({
            "scale": float(scale),
            "relative_residual": float(
                torch.linalg.vector_norm(residual) / baseline_norm),
        })
    best = min(nonlinear_rows, key=lambda row: row["relative_residual"])
    best_residual = function(q0 + best["scale"] * correction).detach()
    return {
        "support": name,
        "free_site_count": int(np.count_nonzero(free_sites)),
        "free_real_coordinate_count": int(n),
        "lsmr_stop_code": int(solution[1]),
        "lsmr_iterations": int(solution[2]),
        "linear_relative_residual": float(
            np.linalg.norm(linear_residual) / baseline_norm),
        "correction_norm_over_state_norm": float(
            torch.linalg.vector_norm(correction)
            / torch.linalg.vector_norm(q0)),
        "nonlinear_scale_rows": nonlinear_rows,
        "best_nonlinear": best,
        "best_nonlinear_residual_diagnostics": residual_diagnostics(
            best_residual, core, shell),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--maxiter", type=int, default=250)
    args = parser.parse_args()
    rank.lock_inertia()

    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    row = next(item for item in evidence["rows"]
               if item["axis"] == args.axis and item["sign"] == args.sign)
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    q0 = torch.tensor(rank.pack(psi, chi), dtype=torch.float64)
    core, shell = site_masks(psi)
    exterior = ~core

    def function(value: torch.Tensor) -> torch.Tensor:
        return recurrence(value, args.axis, args.sign, row["best_dt"],
                          row["best_phase_psi"], row["best_phase_chi"])

    residual0 = function(q0).detach()
    cases = [
        solve_case("C19_first_shell", shell, core, shell,
                   q0, residual0, function,
                   args.maxiter),
        solve_case("all_exterior_sites", exterior, core, shell,
                   q0, residual0, function,
                   args.maxiter),
    ]
    best = min(cases, key=lambda item: item["best_nonlinear"][
        "relative_residual"])
    pass_linear = best["linear_relative_residual"] < 1e-6
    pass_nonlinear = (best["best_nonlinear"]["relative_residual"] < 0.1
                      and best["correction_norm_over_state_norm"] < 0.5)
    result = {
        "status": "V4_EXACT_TRANSLATION_PRECURSOR_LINEAR_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis": args.axis,
        "sign": args.sign,
        "exact_translation_map": (
            "complete joint Psi-chi profile shifted one cell with no runtime "
            "projection, reset, receiver preload, or copied force"),
        "core_values_held_exact_during_inverse_gate": True,
        "dt": row["best_dt"],
        "psi_phase": row["best_phase_psi"],
        "chi_phase": row["best_phase_chi"],
        "baseline_recurrence_norm": float(
            torch.linalg.vector_norm(residual0)),
        "cases": cases,
        "best_case": best,
        "verdict": (
            "CANONICAL_ACTION_HAS_SMALL_NATIVE_PRECURSOR_DIRECTION"
            if pass_linear and pass_nonlinear else
            "CANONICAL_LINEAR_PRECURSOR_DOES_NOT_CLOSE_EXACT_SHIFT"),
        "claim_boundary": (
            "Linearized inverse profile test at one signed axis and the "
            "fixed compact-carrier phases. A favorable result requires a "
            "full nonlinear exact-profile solve, all axes/signs, free "
            "evolution, conservation, and Dirac-current validation. A "
            "failure is not a no-go for a changed reciprocal action."),
    }
    sign_label = "plus" if args.sign > 0 else "minus"
    output = HERE / (
        f"exact_translation_precursor_linear_gate_axis{args.axis}_"
        f"{sign_label}.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "baseline_recurrence_norm": result["baseline_recurrence_norm"],
        "cases": [{
            "support": case["support"],
            "linear_relative_residual": case["linear_relative_residual"],
            "correction_norm_over_state_norm": case[
                "correction_norm_over_state_norm"],
            "best_nonlinear": case["best_nonlinear"],
        } for case in cases],
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
