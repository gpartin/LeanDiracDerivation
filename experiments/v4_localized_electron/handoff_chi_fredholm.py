"""Audit chi solvability at the first reflected-cube handoff.

This script deliberately writes every complex norm as ``real^2 + imag^2``.
PyTorch's nested differentiation through ``abs(z)^2`` can lose second-order
complex curvature, which made the earlier inline Fredholm calculation
unreliable.  The audit computes the canonical chi Hessian and the mixed
derivative forced by the charge-preserving reflected Psi path, then compares
two generic positive research terms:

  * a balanced-density substrate stiffness (``carve``), and
  * a carrier/vacuum interface-orientation stiffness (``interface``).

Neither research term is admitted here.  They are response probes only.  The
output localizes any remaining cokernel obstruction by lattice site so that a
subsequent action change is derived from the missing native channel rather
than guessed from a trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import copy_alternating_handoff as handoff
import localized_stability as native


HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)
WEIGHT_SUM = float(sum(weight for _, weight in native.DIRECTIONS))


def complex_norm_sq(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sum(value.real.square() + value.imag.square(), dim=dim)


def chi_from_real(vector: torch.Tensor, side: int) -> torch.Tensor:
    data = vector.reshape(side, side, side, 2, 3)
    return torch.complex(data[..., 0, :], data[..., 1, :])


def chi_to_real(chi: np.ndarray) -> np.ndarray:
    return np.stack((chi.real, chi.imag), axis=-2).reshape(-1)


def shifted(field: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
    return torch.roll(field, tuple(-value for value in offset), (0, 1, 2))


def canonical_energy(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    p = complex_norm_sq(psi)
    c = complex_norm_sq(chi)
    result = torch.sum(
        (c - native.R**2).square() / 4 + c * p / 2 + p.square() / 4)
    area = torch.linalg.cross(chi, psi, dim=-1)
    for offset, weight in native.DIRECTIONS:
        difference = shifted(area, offset) - area
        result = result + weight * torch.sum(complex_norm_sq(difference)) / (
            2 * native.R**2)
    return result


def carve_energy(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    p = complex_norm_sq(psi)
    c = complex_norm_sq(chi)
    h = ((c - p) / (c + p)).square()
    result = p.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        neighbor_chi = shifted(chi, offset)
        neighbor_h = shifted(h, offset)
        difference = neighbor_chi - chi
        result = result + weight * torch.sum(
            h * neighbor_h * complex_norm_sq(difference)) / (4 * WEIGHT_SUM)
    return result


def interface_energy(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    p = complex_norm_sq(psi)
    result = p.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        neighbor_psi_density = shifted(p, offset)
        neighbor_chi = shifted(chi, offset)
        overlap = torch.sum(chi.conj() * neighbor_chi, dim=-1)
        overlap_sq = overlap.real.square() + overlap.imag.square()
        result = result + weight * torch.sum(
            (p - neighbor_psi_density).square() * overlap_sq
        ) / (2 * native.R**6)
    return result


def psi_path(psi0: torch.Tensor, psi1: torch.Tensor,
             fraction: torch.Tensor) -> torch.Tensor:
    angle = torch.pi * fraction / 2
    return torch.cos(angle) * psi0 + 1j * torch.sin(angle) * psi1


def component_matrices(psi0_np: np.ndarray, psi1_np: np.ndarray,
                       chi0_np: np.ndarray):
    side = psi0_np.shape[0]
    psi0 = torch.as_tensor(psi0_np)
    psi1 = torch.as_tensor(psi1_np)
    x0 = torch.tensor(chi_to_real(chi0_np), requires_grad=True)
    t0 = torch.tensor(0.0, requires_grad=True)

    functions = {
        "canonical": canonical_energy,
        "carve": carve_energy,
        "interface": interface_energy,
    }
    matrices = {}
    for name, energy_function in functions.items():
        def combined(z):
            x, fraction = z[:-1], z[-1]
            psi = psi_path(psi0, psi1, fraction)
            chi = chi_from_real(x, side)
            return energy_function(psi, chi)

        z0 = torch.cat((x0, t0.reshape(1)))
        gradient = torch.autograd.functional.jacobian(
            combined, z0, vectorize=True).detach().numpy()
        hessian = torch.autograd.functional.hessian(
            combined, z0, vectorize=True).detach().numpy()
        matrices[name] = {
            "gradient": gradient[:-1],
            "hessian": hessian[:-1, :-1],
            "forcing": hessian[:-1, -1],
            "mixed_symmetry_defect": float(np.linalg.norm(
                hessian[:-1, -1] - hessian[-1, :-1])),
        }
    return matrices


def fredholm(hessian: np.ndarray, forcing: np.ndarray,
             relative_cutoff: float = 1e-9):
    symmetric = (hessian + hessian.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(values))), 1e-30)
    null = np.abs(values) <= relative_cutoff * scale
    residual = vectors[:, null] @ (vectors[:, null].T @ forcing)
    return {
        "rank": int(np.sum(~null)),
        "nullity": int(np.sum(null)),
        "spectral_scale": scale,
        "cutoff": relative_cutoff * scale,
        "minimum_eigenvalue": float(values[0]),
        "maximum_eigenvalue": float(values[-1]),
        "forcing_norm": float(np.linalg.norm(forcing)),
        "cokernel_norm": float(np.linalg.norm(residual)),
        "cokernel_fraction": float(
            np.linalg.norm(residual) / max(np.linalg.norm(forcing), 1e-30)),
        "residual": residual,
        "eigenvalues": values,
    }


def site_localization(vector: np.ndarray, side: int, mask0: np.ndarray,
                      mask1: np.ndarray):
    field = vector.reshape(side, side, side, 2, 3)
    norms = np.sqrt(np.sum(np.square(field), axis=(-2, -1)))
    rows = []
    for site in np.ndindex((side, side, side)):
        location = (
            "shared_face" if mask0[site] and mask1[site]
            else "source_only" if mask0[site]
            else "destination_only" if mask1[site]
            else "exterior")
        rows.append({
            "site": list(site),
            "location": location,
            "norm": float(norms[site]),
            "real_components": field[site][0].tolist(),
            "imag_components": field[site][1].tolist(),
        })
    rows.sort(key=lambda row: row["norm"], reverse=True)
    totals = {}
    for location in ("shared_face", "source_only", "destination_only", "exterior"):
        selected = [tuple(row["site"]) for row in rows if row["location"] == location]
        totals[location] = float(np.sqrt(sum(norms[site] ** 2 for site in selected)))
    return rows[:16], totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--side", type=int, default=5)
    parser.add_argument("--output", default="handoff_chi_fredholm.json")
    parser.add_argument("--npz-output", default="handoff_chi_fredholm_corrected.npz")
    args = parser.parse_args()

    amplitude_sq = native.R**2 / (2 + 54 / native.R**2)
    psi0, chi0 = native.carrier_fields(args.side, amplitude_sq)
    psi1, _ = handoff.reflected_destination(
        psi0, chi0, args.axis, args.sign)
    mask0 = native.norm2(psi0) > 1e-14
    mask1 = native.norm2(psi1) > 1e-14
    matrices = component_matrices(psi0, psi1, chi0)

    old_path = HERE / "handoff_chi_fredholm_full.npz"
    legacy = np.load(old_path) if old_path.exists() else None
    canonical = matrices["canonical"]
    legacy_comparison = None
    if legacy is not None:
        legacy_hessian = legacy["hessian"]
        legacy_forcing = legacy["forcing"]
        legacy_comparison = {
            "hessian_relative_difference": float(np.linalg.norm(
                canonical["hessian"] - legacy_hessian) /
                max(np.linalg.norm(canonical["hessian"]), 1e-30)),
            "forcing_relative_difference": float(np.linalg.norm(
                canonical["forcing"] - legacy_forcing) /
                max(np.linalg.norm(canonical["forcing"]), 1e-30)),
        }

    cases = {
        "canonical": (canonical["hessian"], canonical["forcing"]),
        "canonical_plus_carve": (
            canonical["hessian"] + matrices["carve"]["hessian"],
            canonical["forcing"] + matrices["carve"]["forcing"]),
        "canonical_plus_interface": (
            canonical["hessian"] + matrices["interface"]["hessian"],
            canonical["forcing"] + matrices["interface"]["forcing"]),
        "canonical_plus_carve_plus_interface": (
            canonical["hessian"] + matrices["carve"]["hessian"]
            + matrices["interface"]["hessian"],
            canonical["forcing"] + matrices["carve"]["forcing"]
            + matrices["interface"]["forcing"]),
    }
    analyses = {name: fredholm(*case) for name, case in cases.items()}
    top_sites, location_norms = site_localization(
        analyses["canonical_plus_carve_plus_interface"]["residual"],
        args.side, mask0, mask1)

    result = {
        "status": "V4_CORRECTED_CHI_HANDOFF_FREDHOLM_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "axis": args.axis,
        "sign": args.sign,
        "side": args.side,
        "amplitude_sq": float(amplitude_sq),
        "complex_second_derivative_convention": "explicit_real_sq_plus_imag_sq",
        "component_gradient_norms": {
            name: float(np.linalg.norm(row["gradient"]))
            for name, row in matrices.items()},
        "component_mixed_symmetry_defects": {
            name: row["mixed_symmetry_defect"] for name, row in matrices.items()},
        "legacy_comparison": legacy_comparison,
        "cases": {
            name: {key: value for key, value in row.items()
                   if key not in ("residual", "eigenvalues")}
            for name, row in analyses.items()},
        "combined_residual_location_norms": location_norms,
        "combined_residual_top_sites": top_sites,
        "verdict": "CORRECTED_HESSIAN_RESPONSE_CLASSIFIED",
        "claim_boundary": (
            "This is an infinitesimal solvability audit along a prescribed "
            "charge-preserving geometric path.  It does not prove a moving "
            "orbit or admit either research action term."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(
        HERE / args.npz_output,
        canonical_hessian=canonical["hessian"],
        canonical_forcing=canonical["forcing"],
        carve_hessian=matrices["carve"]["hessian"],
        carve_forcing=matrices["carve"]["forcing"],
        interface_hessian=matrices["interface"]["hessian"],
        interface_forcing=matrices["interface"]["forcing"],
        combined_residual=analyses[
            "canonical_plus_carve_plus_interface"]["residual"],
    )
    print(json.dumps({
        "status": result["status"],
        "legacy_comparison": legacy_comparison,
        "cases": result["cases"],
        "combined_residual_location_norms": location_norms,
        "combined_residual_top_sites": top_sites[:8],
        "output": str(HERE / args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
