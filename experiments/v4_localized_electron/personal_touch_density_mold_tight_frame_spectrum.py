"""Implicit rest spectrum of tight-frame potential plus density-mold metric."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import density_mold_connection_probe as mold
import density_mold_rest_spectrum as rest
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def tight_frame_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = mold.unpack_tensor(raw, rest.SIDE)
    rho_psi = torch.sum(mold.squared_norm(psi), dim=-1)
    rho_chi = torch.sum(mold.squared_norm(chi), dim=-1)
    carrier = (rho_psi/rank.R**2)**2
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        neighbor_psi = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho_psi = torch.roll(
            rho_psi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho_chi = torch.roll(
            rho_chi, shifts=offset, dims=(0, 1, 2))
        neighbor_carrier = torch.roll(
            carrier, shifts=offset, dims=(0, 1, 2))
        chi_overlap = torch.sum(torch.conj(chi)*neighbor_chi, dim=-1)
        chi_s = torch.real(torch.conj(chi_overlap)*chi_overlap)
        chi_p = rho_chi*neighbor_rho_chi
        result = result+weight*torch.sum(
            carrier*neighbor_carrier*(3*chi_s-chi_p)**2
            )/(4*rank.R**6)
        psi_overlap = torch.sum(torch.conj(psi)*neighbor_psi, dim=-1)
        psi_s = torch.real(torch.conj(psi_overlap)*psi_overlap)
        psi_p = rho_psi*neighbor_rho_psi
        result = result+weight*torch.sum(
            (3*psi_s-psi_p)**2)/(4*rank.R**6)
    return result


BASE_POTENTIAL = rest.potential


def combined_potential(raw: torch.Tensor) -> torch.Tensor:
    return BASE_POTENTIAL(raw)+tight_frame_energy(raw)


def main() -> None:
    rank.lock_inertia()
    rest.potential = combined_potential
    rows = [rest.classify(0.0), rest.classify(mold.ALPHA)]
    candidate = rows[1]
    stable = (candidate["spectral_radius"] is not None
              and candidate["spectral_radius"] <= 1+1e-7)
    result = {
        "status": "V4_TIGHT_FRAME_DENSITY_MOLD_REST_SPECTRUM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "position_action": "unit chi and Psi carrier tight-frame bonds",
        "kinetic_action": (
            "B/2[kappa|chiDot|^2+|A19(chiDot+s rhoPsiDot chi/"
            "(2(rhoChi+kappa chi0^2)))/24|^2]"),
        "rows": rows,
        "verdict": (
            "COMBINED_IMPLICIT_REST_SPECTRUM_PASS" if stable
            else "COMBINED_IMPLICIT_REST_SPECTRUM_FAIL"),
        "claim_boundary": (
            "Complete side-three implicit two-layer rest spectrum. A pass "
            "still requires side-five/multishift, nonlinear conservation, "
            "free relative-periodic motion, and Dirac-current closure."),
    }
    (HERE/"personal_touch_density_mold_tight_frame_spectrum.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
