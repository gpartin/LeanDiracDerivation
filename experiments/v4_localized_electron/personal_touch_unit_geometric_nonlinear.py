"""Unrestricted nonlinear Verlet gate for the unit geometric action.

The action is the bare canonical V40 action plus the five unit square-face
identities and four unit density-activated C19 bond identities.  All added
terms are position-only, so the native centered Verlet update remains valid.
No state projection, reset, held background, or fitted coefficient is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5
ACTION_KIND = "unit"


def unpack(raw: torch.Tensor):
    return box.unpack(raw, SIDE)


def norm2(value: torch.Tensor, dimensions=(-1,)) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=dimensions))


def shift(value: torch.Tensor, axis: int) -> torch.Tensor:
    offset = tuple(1 if entry == axis else 0 for entry in range(3))
    return torch.roll(value, shifts=offset, dims=(0, 1, 2))


def outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left[..., :, None]*right[..., None, :]


def canonical_potential(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    rho_psi, rho_chi = norm2(psi), norm2(chi)
    local = torch.sum(0.25*(rho_chi-small.CHI0**2)**2
                      + 0.5*rho_chi*rho_psi + 0.25*rho_psi**2)
    area = torch.linalg.cross(chi, psi, dim=-1)
    bond = raw.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        difference = area-torch.roll(
            area, shifts=offset, dims=(0, 1, 2))
        bond += weight*torch.sum(norm2(difference))
    return local+bond/(2*small.CHI0**2)


def face_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    w = torch.linalg.cross(chi, psi, dim=-1)/small.CHI0
    result = raw.new_zeros(())
    for first, second in ((0, 1), (0, 2), (1, 2)):
        wa, wb = shift(w, first), shift(w, second)
        wab = shift(wa, second)
        diagonal = outer(w, wab)-outer(wa, wb)
        symmetric = (diagonal+diagonal.transpose(-1, -2))/2
        antisymmetric = (diagonal-diagonal.transpose(-1, -2))/2
        reversed_diagonal = outer(w, wab)-outer(wb, wa)
        conjugate_diagonal = (
            outer(w, torch.conj(wab))-outer(wa, torch.conj(wb)))
        da, db = wa-w, wb-w
        edge = outer(da, db)-outer(db, da)
        for defect in (symmetric, antisymmetric, reversed_diagonal,
                       conjugate_diagonal, edge):
            result += torch.sum(norm2(defect, (-2, -1)))/2
    return result


def activated_bond_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    rho_psi, rho_chi = norm2(psi), norm2(chi)
    vacuum = ((rho_chi-rho_psi)/small.CHI0**2)**2
    carrier = (rho_psi/small.CHI0**2)**2
    result = raw.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        neighbor_psi = torch.roll(psi, shifts=offset, dims=(0, 1, 2))
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho_psi = torch.roll(
            rho_psi, shifts=offset, dims=(0, 1, 2))
        neighbor_rho_chi = torch.roll(
            rho_chi, shifts=offset, dims=(0, 1, 2))
        neighbor_vacuum = torch.roll(
            vacuum, shifts=offset, dims=(0, 1, 2))
        neighbor_carrier = torch.roll(
            carrier, shifts=offset, dims=(0, 1, 2))
        chi_overlap = torch.sum(torch.conj(chi)*neighbor_chi, dim=-1)
        chi_s = torch.real(torch.conj(chi_overlap)*chi_overlap)
        chi_p = rho_chi*neighbor_rho_chi
        result += weight*torch.sum(
            vacuum*neighbor_vacuum*(chi_p-chi_s))/(4*small.CHI0**2)
        interface = vacuum*neighbor_carrier+carrier*neighbor_vacuum
        result += weight*torch.sum(interface*chi_s)/(4*small.CHI0**2)
        result += weight*torch.sum(
            carrier*neighbor_carrier*(3*chi_s-chi_p)**2
            )/(4*small.CHI0**6)
        psi_overlap = torch.sum(torch.conj(psi)*neighbor_psi, dim=-1)
        psi_s = torch.real(torch.conj(psi_overlap)*psi_overlap)
        psi_p = rho_psi*neighbor_rho_psi
        result += weight*torch.sum(
            (3*psi_s-psi_p)**2)/(4*small.CHI0**6)
    return result


def carrier_vacuum_interface_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    rho_psi, rho_chi = norm2(psi), norm2(chi)
    vacuum = ((rho_chi-rho_psi)/small.CHI0**2)**2
    carrier = (rho_psi/small.CHI0**2)**2
    result = raw.new_zeros(())
    for offset, weight in native.DIRECTIONS:
        neighbor_chi = torch.roll(chi, shifts=offset, dims=(0, 1, 2))
        neighbor_vacuum = torch.roll(
            vacuum, shifts=offset, dims=(0, 1, 2))
        neighbor_carrier = torch.roll(
            carrier, shifts=offset, dims=(0, 1, 2))
        chi_overlap = torch.sum(torch.conj(chi)*neighbor_chi, dim=-1)
        chi_s = torch.real(torch.conj(chi_overlap)*chi_overlap)
        interface = vacuum*neighbor_carrier+carrier*neighbor_vacuum
        result += weight*torch.sum(interface*chi_s)/(4*small.CHI0**2)
    return result


def carrier_tight_frame_energy(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = unpack(raw)
    rho_psi, rho_chi = norm2(psi), norm2(chi)
    carrier = (rho_psi/small.CHI0**2)**2
    result = raw.new_zeros(())
    for offset, weight in native.DIRECTIONS:
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
        result += weight*torch.sum(
            carrier*neighbor_carrier*(3*chi_s-chi_p)**2
            )/(4*small.CHI0**6)
        psi_overlap = torch.sum(torch.conj(psi)*neighbor_psi, dim=-1)
        psi_s = torch.real(torch.conj(psi_overlap)*psi_overlap)
        psi_p = rho_psi*neighbor_rho_psi
        result += weight*torch.sum(
            (3*psi_s-psi_p)**2)/(4*small.CHI0**6)
    return result


def total_potential(raw: torch.Tensor) -> torch.Tensor:
    if ACTION_KIND == "reduced_interface":
        return canonical_potential(raw)+carrier_vacuum_interface_energy(raw)
    if ACTION_KIND == "reduced_tight_frame":
        return canonical_potential(raw)+carrier_tight_frame_energy(raw)
    return canonical_potential(raw)+face_energy(raw)+activated_bond_energy(raw)


def pack(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.real(psi), torch.imag(psi),
                        torch.real(chi), torch.imag(chi)), dim=-2).reshape(-1)


def rotate(raw: torch.Tensor, psi_phase: float, chi_phase: float) -> torch.Tensor:
    psi, chi = unpack(raw)
    pp = torch.exp(torch.tensor(1j*psi_phase, dtype=torch.complex128,
                                device=raw.device))
    pc = torch.exp(torch.tensor(1j*chi_phase, dtype=torch.complex128,
                                device=raw.device))
    return pack(pp*psi, pc*chi)


def masses(device) -> torch.Tensor:
    shaped = torch.ones((SIDE, SIDE, SIDE, 4, 3), dtype=torch.float64,
                        device=device)
    shaped[..., 2:, :] = small.B
    return shaped.reshape(-1)


def noether_charge(left: torch.Tensor, right: torch.Tensor,
                   field: str, dt: float) -> float:
    psi, chi = unpack(right)
    zp, zc = torch.zeros_like(psi), torch.zeros_like(chi)
    generator = pack(1j*psi if field == "Psi" else zp,
                     1j*chi if field == "chi" else zc)
    return float(torch.dot(
        masses(right.device)*(right-left)/dt, generator).detach().cpu())


def interval_energy(left: torch.Tensor, right: torch.Tensor,
                    dt: float) -> float:
    velocity = (right-left)/dt
    kinetic = 0.5*torch.sum(masses(right.device)*velocity**2)
    potential = (total_potential(left)+total_potential(right))/2
    return float((kinetic+potential).detach().cpu())


def aligned_metrics(raw: torch.Tensor, reference: torch.Tensor,
                    carrier_mask: np.ndarray) -> dict:
    psi, chi = unpack(raw)
    psi0, chi0 = unpack(reference)
    psi_phase = torch.angle(torch.sum(torch.conj(psi0)*psi))
    chi_phase = torch.angle(torch.sum(torch.conj(chi0)*chi))
    aligned = rotate(raw, -float(psi_phase), -float(chi_phase))
    aligned_psi, _aligned_chi = unpack(aligned)
    psi_projector = outer(psi, torch.conj(psi))
    chi_projector = outer(chi, torch.conj(chi))
    psi0_projector = outer(psi0, torch.conj(psi0))
    chi0_projector = outer(chi0, torch.conj(chi0))
    area = torch.linalg.cross(chi, psi, dim=-1)
    area0 = torch.linalg.cross(chi0, psi0, dim=-1)
    area_projector = outer(area, torch.conj(area))
    area0_projector = outer(area0, torch.conj(area0))
    rho_chi = norm2(chi)
    rho_chi0 = norm2(chi0)
    mask = torch.tensor(carrier_mask, device=raw.device)
    overlap = torch.abs(torch.sum(
        torch.conj(psi0[mask])*aligned_psi[mask]))**2
    fidelity = overlap/(torch.sum(torch.abs(psi0[mask])**2)
                        * torch.sum(torch.abs(aligned_psi[mask])**2))
    exterior = torch.sum(torch.abs(aligned_psi[~mask])**2)/torch.sum(
        torch.abs(aligned_psi)**2)
    return {
        "relative_shape_residual": float(
            torch.linalg.vector_norm(aligned-reference).detach().cpu()
            / torch.linalg.vector_norm(reference).detach().cpu()),
        "Psi_carrier_fidelity": float(fidelity.detach().cpu()),
        "Psi_exterior_fraction": float(exterior.detach().cpu()),
        "Psi_projector_relative_residual": float(
            torch.linalg.vector_norm(psi_projector-psi0_projector).detach().cpu()
            / max(float(torch.linalg.vector_norm(psi0_projector).detach().cpu()), 1e-300)),
        "chi_projector_relative_residual": float(
            torch.linalg.vector_norm(chi_projector-chi0_projector).detach().cpu()
            / max(float(torch.linalg.vector_norm(chi0_projector).detach().cpu()), 1e-300)),
        "area_projector_relative_residual": float(
            torch.linalg.vector_norm(area_projector-area0_projector).detach().cpu()
            / max(float(torch.linalg.vector_norm(area0_projector).detach().cpu()), 1e-300)),
        "chi_density_relative_residual": float(
            torch.linalg.vector_norm(rho_chi-rho_chi0).detach().cpu()
            / max(float(torch.linalg.vector_norm(rho_chi0).detach().cpu()), 1e-300)),
        "removed_Psi_phase": float(psi_phase.detach().cpu()),
        "removed_chi_phase": float(chi_phase.detach().cpu()),
    }


def main() -> None:
    global ACTION_KIND
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--dt-scale", type=float, default=1.0)
    parser.add_argument("--clock-fraction", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-force", action="store_true")
    parser.add_argument(
        "--action",
        choices=("unit", "reduced_interface", "reduced_tight_frame"),
        default="unit")
    parser.add_argument("--output", default="personal_touch_unit_geometric_nonlinear.json")
    args = parser.parse_args()
    ACTION_KIND = args.action
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dt = small.DT*args.dt_scale
    raw = np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"]
    reference = torch.tensor(raw, dtype=torch.float64, device=device)
    theta = float(np.arccos(1-dt**2*small.CHI0**2/2))
    current = reference.clone()
    previous = rotate(reference, -theta, -args.clock_fraction*theta)
    psi0, _chi0 = box.unpack(torch.tensor(raw), SIDE)
    carrier_mask = (torch.sum(torch.abs(psi0)**2, dim=-1) > 1e-12).numpy()
    mass = masses(device)
    force_function = torch.func.grad(total_potential)
    try:
        if not args.compile_force:
            raise RuntimeError("compiled force not requested")
        force_function = torch.compile(force_function, fullgraph=False)
        compiled = True
    except Exception:
        compiled = False
    initial_qpsi = noether_charge(previous, current, "Psi", dt)
    initial_qchi = noether_charge(previous, current, "chi", dt)
    initial_energy = interval_energy(previous, current, dt)
    steps_per_cycle = int(round(2*np.pi/theta))
    total_steps = args.cycles*steps_per_cycle
    samples = []
    max_shape, min_fidelity, max_exterior = 0.0, 1.0, 0.0
    max_psi_projector = 0.0
    max_chi_projector = 0.0
    max_area_projector = 0.0
    max_chi_density = 0.0
    finite = True
    for index in range(1, total_steps+1):
        point = current.detach().requires_grad_(True)
        gradient = force_function(point)
        following = 2*current-previous-dt**2*gradient.detach()/mass
        previous, current = current, following
        finite = finite and bool(torch.all(torch.isfinite(current)))
        if index == 1 or index % steps_per_cycle == 0 or index == total_steps:
            metrics = aligned_metrics(current, reference, carrier_mask)
            max_shape = max(max_shape, metrics["relative_shape_residual"])
            min_fidelity = min(min_fidelity, metrics["Psi_carrier_fidelity"])
            max_exterior = max(max_exterior, metrics["Psi_exterior_fraction"])
            max_psi_projector = max(
                max_psi_projector,
                metrics["Psi_projector_relative_residual"])
            max_chi_projector = max(
                max_chi_projector,
                metrics["chi_projector_relative_residual"])
            max_area_projector = max(
                max_area_projector,
                metrics["area_projector_relative_residual"])
            max_chi_density = max(
                max_chi_density,
                metrics["chi_density_relative_residual"])
            qpsi = noether_charge(previous, current, "Psi", dt)
            qchi = noether_charge(previous, current, "chi", dt)
            energy = interval_energy(previous, current, dt)
            samples.append({
                "step": index,
                "cycle": index/steps_per_cycle,
                **metrics,
                "Psi_charge_relative_drift": abs(qpsi-initial_qpsi)/max(
                    abs(initial_qpsi), 1.0),
                "chi_charge_relative_drift": abs(qchi-initial_qchi)/max(
                    abs(initial_qchi), 1.0),
                "interval_energy_relative_drift": abs(
                    energy-initial_energy)/max(abs(initial_energy), 1.0),
            })
        if not finite:
            break
    result = {
        "status": "V4_UNIT_GEOMETRIC_NONLINEAR_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "research_action": ACTION_KIND,
        "device": str(device),
        "compiled_force": compiled,
        "cycles_requested": args.cycles,
        "steps_per_cycle": steps_per_cycle,
        "steps_completed": index,
        "dt": dt,
        "clock_fraction": args.clock_fraction,
        "finite": finite,
        "maximum_relative_shape_residual": max_shape,
        "minimum_Psi_carrier_fidelity": min_fidelity,
        "maximum_Psi_exterior_fraction": max_exterior,
        "maximum_Psi_projector_relative_residual": max_psi_projector,
        "maximum_chi_projector_relative_residual": max_chi_projector,
        "maximum_area_projector_relative_residual": max_area_projector,
        "maximum_chi_density_relative_residual": max_chi_density,
        "samples": samples,
        "verdict": (
            "UNIT_GEOMETRIC_NONLINEAR_PRELIMINARY_PASS"
            if finite and min_fidelity > 0.99 and max_exterior < 1e-4
            else "UNIT_GEOMETRIC_NONLINEAR_FAIL"),
        "claim_boundary": (
            "Unrestricted nonlinear centered-Verlet evolution with chi-clock "
            f"fraction {args.clock_fraction}. A zero-clock 100-cycle result "
            "establishes only the exact rest orbit. A nonzero conserved chi "
            "charge requires its own re-solved equilibrium before it can be "
            "used as a perturbation gate. Motion and Dirac transport remain "
            "separate acceptance requirements."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
