"""Resolve the unit-multiplier Jordan sector of the native C27/gyro action.

The full Floquet spectrum has a highly degenerate multiplier at one.  Raw
100-cycle norms therefore mix shape deformation with conserved phase/frequency
and translation-family drift.  This diagnostic constructs ker(M-I) and
ker((M-I)^2) from the actual two-layer monodromy and tests the complement.

The numerical generalized kernel is a diagnostic oracle, not an admissible
runtime projection.  A passing quotient must subsequently be identified with
independently derived Noether and translation tangents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_linear_100cycle as longtime
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
TOLERANCES = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
CHECKPOINT_CYCLES = (1, 10, 100, 1000)


def null_basis(matrix: np.ndarray, relative_tolerance: float):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = relative_tolerance * max(singular[0], 1e-300)
    rank = int(np.sum(singular > threshold))
    return vh[rank:].T, singular, threshold


def field_decomposition(state: np.ndarray, occupied: np.ndarray):
    dimension = state.size // 2
    totals = {
        "psi_carrier": 0.0,
        "psi_exterior": 0.0,
        "chi_carrier": 0.0,
        "chi_exterior": 0.0,
    }
    for raw in (state[:dimension], state[dimension:]):
        psi, chi = relative.unpack_raw(raw, spectrum.SIDE)
        totals["psi_carrier"] += float(np.sum(abs(psi[occupied])**2))
        totals["psi_exterior"] += float(np.sum(abs(psi[~occupied])**2))
        totals["chi_carrier"] += float(np.sum(abs(chi[occupied])**2))
        totals["chi_exterior"] += float(np.sum(abs(chi[~occupied])**2))
    total = max(sum(totals.values()), 1e-300)
    return {name: value / total for name, value in totals.items()}


def observable_vector(state: np.ndarray, rest_psi: np.ndarray,
                      rest_chi: np.ndarray):
    """Linearized native densities and orientation on both committed layers."""
    dimension = state.size // 2
    rows = []
    for raw in (state[:dimension], state[dimension:]):
        dpsi, dchi = relative.unpack_raw(raw, spectrum.SIDE)
        drho_psi = 2 * np.real(np.sum(np.conj(rest_psi) * dpsi, axis=-1))
        drho_chi = 2 * np.real(np.sum(np.conj(rest_chi) * dchi, axis=-1))
        dw = (np.cross(dchi, rest_psi) + np.cross(rest_chi, dpsi))
        rows.extend((drho_psi.ravel(), drho_chi.ravel(),
                     dw.real.ravel(), dw.imag.ravel()))
    return np.concatenate(rows)


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    matrix = longtime.monodromy(True, layers, theta)
    identity = np.eye(matrix.shape[0])
    defect = matrix - identity
    defect2 = defect @ defect
    vectors = longtime.perturbations(layers[1])
    rest_psi, rest_chi = relative.unpack_raw(
        layers[1].detach().numpy(), spectrum.SIDE)
    occupied = np.sum(abs(rest_psi)**2, axis=-1) > 1e-12
    controls = {}
    for name, vector in vectors.items():
        controls[f"{name}:zero_velocity"] = np.r_[vector, vector]
        controls[f"{name}:velocity"] = np.r_[np.zeros_like(vector), -vector]
    steps_per_cycle = int(round(2 * np.pi / theta))
    cycle = np.linalg.matrix_power(matrix, steps_per_cycle)
    cycle_powers = {
        value: np.linalg.matrix_power(cycle, value)
        for value in CHECKPOINT_CYCLES
    }

    rows = []
    for tolerance in TOLERANCES:
        eigenspace, singular1, threshold1 = null_basis(defect, tolerance)
        generalized, singular2, threshold2 = null_basis(defect2, tolerance)
        residuals = {}
        final_complements = {}
        for name, initial in controls.items():
            initial_observable_norm = np.linalg.norm(observable_vector(
                initial, rest_psi, rest_chi))
            samples = {}
            for cycle_count, power in cycle_powers.items():
                final = power @ initial
                complement = final - generalized @ (generalized.T @ final)
                if cycle_count == 100:
                    final_complements[name] = complement
                samples[str(cycle_count)] = {
                    "raw_amplification": float(
                        np.linalg.norm(final)
                        / max(np.linalg.norm(initial), 1e-300)),
                    "generalized_quotient_amplification": float(
                        np.linalg.norm(complement)
                        / max(np.linalg.norm(initial), 1e-300)),
                    "native_observable_amplification": float(
                        np.linalg.norm(observable_vector(
                            complement, rest_psi, rest_chi))
                        / max(initial_observable_norm, 1e-300)),
                }
            residuals[name] = samples
        worst_name = max(
            residuals,
            key=lambda name: residuals[name]["100"][
                "generalized_quotient_amplification"])
        rows.append({
            "relative_tolerance": tolerance,
            "defect_threshold": float(threshold1),
            "defect2_threshold": float(threshold2),
            "eigenspace_dimension": int(eigenspace.shape[1]),
            "generalized_eigenspace_dimension": int(generalized.shape[1]),
            "smallest_defect_singular_values": [
                float(value) for value in singular1[-20:]],
            "smallest_defect2_singular_values": [
                float(value) for value in singular2[-20:]],
            "controls": residuals,
            "maximum_generalized_quotient_amplification": float(max(
                value["100"]["generalized_quotient_amplification"]
                for value in residuals.values())),
            "maximum_quotient_amplification_by_cycle": {
                str(cycle_count): float(max(
                    value[str(cycle_count)][
                        "generalized_quotient_amplification"]
                    for value in residuals.values()))
                for cycle_count in CHECKPOINT_CYCLES
            },
            "maximum_native_observable_amplification_by_cycle": {
                str(cycle_count): float(max(
                    value[str(cycle_count)][
                        "native_observable_amplification"]
                    for value in residuals.values()
                    if value[str(cycle_count)][
                        "native_observable_amplification"] < 1e250))
                for cycle_count in CHECKPOINT_CYCLES
            },
            "worst_100_cycle_control": worst_name,
            "worst_100_cycle_field_fractions": field_decomposition(
                final_complements[worst_name], occupied),
        })

    stable_rows = [row for row in rows
                   if row["maximum_generalized_quotient_amplification"] < 10]
    result = {
        "status": "V4_C27_BODY_GYRO_JORDAN_QUOTIENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "steps_per_cycle": steps_per_cycle,
        "checkpoint_cycles": list(CHECKPOINT_CYCLES),
        "tolerances": list(TOLERANCES),
        "rows": rows,
        "bounded_after_generalized_unit_quotient": bool(stable_rows),
        "verdict": (
            "C27_BODY_GYRO_SECULAR_GROWTH_IS_UNIT_JORDAN_SECTOR"
            if stable_rows else
            "C27_BODY_GYRO_HAS_NONJORDAN_SHAPE_GROWTH"),
        "claim_boundary": (
            "Numerical invariant-subspace diagnosis only. The generalized "
            "kernel cannot be projected out in evolution and must be matched "
            "to independently derived phase, charge, frequency, vacuum-gauge, "
            "and translation tangents before a stability claim."),
    }
    output = HERE / "c27_body_gyro_jordan_quotient.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rows": [{
            "tolerance": row["relative_tolerance"],
            "eigenspace_dimension": row["eigenspace_dimension"],
            "generalized_dimension": row["generalized_eigenspace_dimension"],
            "maximum_quotient_amplification": row[
                "maximum_generalized_quotient_amplification"],
        } for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
