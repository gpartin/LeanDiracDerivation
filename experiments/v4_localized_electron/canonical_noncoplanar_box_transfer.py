"""Transfer the side-three canonical branch to ordinary larger-box vacuum."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as source
import localized_stability as native


HERE = Path(__file__).resolve().parent


def pack(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    return np.stack((psi.real, psi.imag, chi.real, chi.imag), axis=-2).ravel()


def unpack(raw: torch.Tensor, side: int):
    shaped = raw.reshape(side, side, side, 4, 3)
    return (torch.complex(shaped[..., 0, :], shaped[..., 1, :]),
            torch.complex(shaped[..., 2, :], shaped[..., 3, :]))


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=-1))


def potential(raw: torch.Tensor, side: int) -> torch.Tensor:
    psi, chi = unpack(raw, side)
    rp, rc = squared_norm(psi), squared_norm(chi)
    local = torch.sum(0.25*(rc-source.CHI0**2)**2
                      + 0.5*rc*rp + 0.25*rp**2)
    w = torch.linalg.cross(chi, psi, dim=-1)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        difference = w - torch.roll(w, shifts=offset, dims=(0, 1, 2))
        bond += weight*torch.sum(squared_norm(difference))
    return local + bond/(2*source.CHI0**2)


def interval(left: torch.Tensor, right: torch.Tensor, side: int):
    lp, lc = unpack(left, side)
    rp, rc = unpack(right, side)
    kinetic = 0.5*torch.sum(squared_norm((rp-lp)/source.DT))
    kinetic += source.B/2*torch.sum(squared_norm((rc-lc)/source.DT))
    return source.DT*(kinetic-(potential(left, side)+potential(right, side))/2)


def phase_rotate(raw: torch.Tensor, phase: complex, side: int):
    psi, chi = unpack(raw, side)
    p = torch.tensor(phase, dtype=torch.complex128)
    return torch.stack((torch.real(p*psi), torch.imag(p*psi),
                        torch.real(chi), torch.imag(chi)), dim=-2).reshape(-1)


def residual(raw: np.ndarray, side: int, phase: complex) -> np.ndarray:
    value = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    action = (interval(phase_rotate(value, np.conjugate(phase), side),
                       value, side)
              + interval(value, phase_rotate(value, phase, side), side))
    return torch.autograd.grad(action, value)[0].detach().numpy()


def embed(rest3: np.ndarray, state3: np.ndarray, side: int) -> np.ndarray:
    psi3, chi3 = source.unpack(state3)
    rest_psi3, rest_chi3 = source.unpack(rest3)
    psi, chi = native.carrier_fields(side, source.AMPLITUDE_SQ)
    origin = side//2-1
    for corner in itertools.product((0, 1), repeat=3):
        site3 = corner
        site = tuple(origin+bit for bit in corner)
        psi[site] += psi3[site3]-rest_psi3[site3]
        chi[site] += chi3[site3]-rest_chi3[site3]
    return pack(psi, chi)


def main() -> None:
    archive = np.load(HERE / "canonical_noncoplanar_augmented_branch.npz")
    rest3 = archive["rest"]
    fractions = archive["fractions"]
    states = archive["states"]
    theta = float(np.arccos(1-source.DT**2*source.CHI0**2/2))
    phase = np.exp(1j*theta)
    rows = []
    for side in (3, 4, 5):
        rest_psi, rest_chi = native.carrier_fields(side, source.AMPLITUDE_SQ)
        rest = pack(rest_psi, rest_chi)
        rest_residual = residual(rest, side, phase)
        for fraction, state3 in zip(fractions, states, strict=True):
            state = state3 if side == 3 else embed(rest3, state3, side)
            value = residual(state, side, phase)
            psi, _chi = unpack(torch.tensor(state), side)
            occupied = torch.sum(torch.abs(psi)**2, dim=-1) > 1e-12
            rows.append({
                "side": side,
                "fraction_of_rest_norm": float(fraction),
                "baseline_rest_residual_rms": float(
                    np.linalg.norm(rest_residual)/np.sqrt(rest_residual.size)),
                "transferred_residual_rms": float(
                    np.linalg.norm(value)/np.sqrt(value.size)),
                "transferred_residual_over_baseline": float(
                    np.linalg.norm(value)/max(np.linalg.norm(rest_residual),
                                              1e-300)),
                "occupied_site_count": int(torch.sum(occupied)),
            })
    localized = [row for row in rows if row["side"] > 3
                 and row["transferred_residual_rms"] < 1e-9]
    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_BOX_TRANSFER",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "verdict": (
            "NONCOPLANAR_BRANCH_TRANSFERS_TO_ORDINARY_VACUUM"
            if localized else
            "NONCOPLANAR_BRANCH_IS_SMALL_PERIOD_TEXTURE_NOT_LOCALIZED_BRANCH"),
        "claim_boundary": (
            "Direct core transplant into canonical larger-box vacuum. Failure "
            "does not exclude a re-solved branch with a self-consistent halo."),
    }
    output = HERE / "canonical_noncoplanar_box_transfer.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
