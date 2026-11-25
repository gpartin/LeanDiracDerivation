"""Unit geometric action on the bare canonical V40 two-field baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_transfer_oscillator_implicit_update as update
import canonical_transfer_oscillator_spectrum as spectrum
import personal_touch_activated_bond_exact_scan as activated_names
import personal_touch_face_psd_cone as cone
import personal_touch_face_shift_invert_scan as face_scan
import personal_touch_vacuum_chi_exact_scan as exact


HERE = Path(__file__).resolve().parent
CACHE = HERE/"personal_touch_bare_canonical_matrices.npz"
SHIFTS = (
    1.00000005+0j,
    1.000001+1e-6j,
    1.000001+5e-6j,
    1.000001+1e-5j,
    1.000005+5e-5j,
    1.00001+1e-4j,
    1.0001+1e-3j,
)


def derive_bare():
    update.STRENGTH = 0.0
    update.EVEN_STRENGTH = 0.0
    update.PARALLEL_STRENGTH = 0.0
    update.DENSITY_BALANCE_STRENGTH = 0.0
    update.SPECTRAL_AREA_STRENGTH = 0.0
    update.LOOP_CURVATURE_STRENGTH = 0.0
    update.FACE_FACTORIZATION_STRENGTH = 0.0
    update.FACE_CONJUGATE_FACTORIZATION_STRENGTH = 0.0
    update.FACE_EDGE_COMMUTATOR_STRENGTH = 0.0
    rest = np.load(
        HERE/"canonical_noncoplanar_large_box_solvability.npz")["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    current = torch.tensor(rest, dtype=torch.float64)
    previous = box.phase_rotate(
        current, np.exp(-1j*theta), cone.SIDE).detach()
    following = box.phase_rotate(
        current, np.exp(1j*theta), cone.SIDE).detach()
    blocks = []
    for argument in range(3):
        def function(value, index=argument):
            layers = [previous, current, following]
            layers[index] = value
            return update.del_residual(*layers)
        blocks.append(torch.autograd.functional.jacobian(
            function, (previous, current, following)[argument],
            vectorize=True).detach().numpy())
    rotation = spectrum.rotation_matrix(rest.size, theta)
    minus, zero, plus = blocks[0]@rotation.T, blocks[1], blocks[2]@rotation
    mass = -0.5*small.DT*(plus+minus)
    gyro = -(plus-minus)
    stiffness = -(plus+zero+minus)/small.DT
    result = {
        "rest": rest,
        "mass": (mass+mass.T)/2,
        "gyro": (gyro-gyro.T)/2,
        "stiffness": (stiffness+stiffness.T)/2,
    }
    np.savez_compressed(CACHE, **result)
    return result


def main() -> None:
    bare = ({key: value for key, value in np.load(CACHE).items()}
            if CACHE.exists() else derive_bare())
    faces = np.load(cone.CACHE)
    activated = np.load(
        HERE/"personal_touch_activated_bond_basis_v2_cache.npz")
    candidate = bare["stiffness"].copy()
    for name in face_scan.NAMES:
        candidate += faces[f"hessian_{name}"]
    for name in activated_names.ACTIVATED_NAMES:
        candidate += activated[f"hessian_{name}"]
    exact.REQUESTED = 2
    rows, collected = [], []
    for shift in SHIFTS:
        exact.SIGMA = shift
        values, converged = exact.targeted_spectrum(
            bare["mass"], bare["gyro"], candidate)
        collected.extend(values.tolist())
        row = {
            "shift": {"real": float(shift.real), "imag": float(shift.imag)},
            "converged": converged and len(values) == exact.REQUESTED,
            "values": [
                {"real": float(value.real), "imag": float(value.imag),
                 "modulus": float(abs(value))} for value in values],
        }
        rows.append(row)
        print(shift, row["converged"], row["values"], flush=True)
    unique = []
    for value in collected:
        if not any(abs(value-existing) < 1e-8 for existing in unique):
            unique.append(value)
    moduli = np.abs(unique)
    result = {
        "status": "V4_UNIT_GEOMETRIC_ACTION_ON_BARE_CANONICAL",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "baseline": "bare canonical V40 U_local+U_area; no transfer or density scaffolding",
        "coefficient_rule": "unit weight for every normalized positive geometric identity",
        "rows": rows,
        "maximum_resolved_modulus": float(np.max(moduli)),
        "count_resolved_above_1_plus_1e_minus_7": int(
            np.sum(moduli > 1+1e-7)),
        "verdict": (
            "UNIT_GEOMETRIC_BARE_CANONICAL_RESOLVED_BAND_PASS"
            if all(row["converged"] for row in rows)
            and not np.any(moduli > 1+1e-7)
            else "UNIT_GEOMETRIC_BARE_CANONICAL_RETAINS_GROWTH"
            if np.any(moduli > 1+1e-7) else
            "UNIT_GEOMETRIC_BARE_CANONICAL_INCONCLUSIVE"),
        "claim_boundary": (
            "Exact modes nearest seven complex shifts. No full contour, "
            "nonlinear, conservation, motion, or Dirac-regression claim."),
    }
    (HERE/"personal_touch_unit_geometric_canonical.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "maximum_resolved_modulus": result["maximum_resolved_modulus"],
        "count_resolved_above_1_plus_1e_minus_7": result[
            "count_resolved_above_1_plus_1e_minus_7"],
        "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
