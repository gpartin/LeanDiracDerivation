"""Derive the candidate beta grading from the native Hodge configuration image.

The unactivated relative composite has two complex configuration channels per
site.  If those channels are the positive-beta half of a Dirac field, their
orthogonal projector P must be local and beta=2P-I must anticommute with the
three C19 Hodge spatial generators.  This script checks those identities on
the complete periodic five-cube field, without supplying a Dirac beta matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cube_incidence_hodge_readout_bridge as readout
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
SITE_REAL = 8
FIELD_REAL = SIDE**3*SITE_REAL


def main() -> None:
    rest = np.load(canonical.CACHE)["rest"]
    historical = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    mapped = readout.composite_jacobian(
        rest, historical["plus"], activated=False)
    u, singular, _vh = np.linalg.svd(mapped, full_matrices=False)
    threshold = 1e-10*max(float(singular[0]), 1e-30)
    rank = int(np.sum(singular > threshold))
    q = u[:, :rank]
    projector = q@q.T
    identity = np.eye(FIELD_REAL)
    beta = 2*projector-identity

    # Spatial locality of the derived grading.
    blocks = projector.reshape(
        SIDE**3, SITE_REAL, SIDE**3, SITE_REAL).transpose(0, 2, 1, 3)
    diagonal = np.asarray([blocks[index, index]
                           for index in range(SIDE**3)])
    offsite = blocks.copy()
    for index in range(SIDE**3):
        offsite[index, index] = 0
    mean_local = np.mean(diagonal, axis=0)

    hodge_operator = readout.hodge_on_columns(identity)
    anticommutator = beta@hodge_operator+hodge_operator@beta
    same_grade = projector@hodge_operator@projector
    complement = identity-projector
    opposite_same_grade = complement@hodge_operator@complement

    # Internal local beta and its complex-linearity.
    local_beta = 2*mean_local-np.eye(SITE_REAL)
    complex_structure = np.zeros((SITE_REAL, SITE_REAL))
    complex_structure[0::2, 1::2] = -np.eye(SITE_REAL//2)
    complex_structure[1::2, 0::2] = np.eye(SITE_REAL//2)
    local_values = np.linalg.eigvalsh((local_beta+local_beta.T)/2)

    scale = max(np.linalg.norm(hodge_operator), 1e-30)
    result = {
        "status": "V4_C19_HODGE_BETA_GRADING_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "configuration_rank": rank,
        "derived_beta": "2 projector(image d(tau,W)) - identity",
        "projector_checks": {
            "idempotence_defect": float(np.linalg.norm(
                projector@projector-projector)),
            "symmetry_defect": float(np.linalg.norm(
                projector-projector.T)),
            "offsite_block_fraction": float(
                np.linalg.norm(offsite)/np.linalg.norm(projector)),
            "local_block_translation_variation": float(
                np.linalg.norm(diagonal-mean_local[None, ...])
                / np.linalg.norm(diagonal)),
        },
        "beta_checks": {
            "involution_defect": float(np.linalg.norm(beta@beta-identity)),
            "symmetry_defect": float(np.linalg.norm(beta-beta.T)),
            "hodge_anticommutator_relative": float(
                np.linalg.norm(anticommutator)/scale),
            "positive_same_grade_spatial_fraction": float(
                np.linalg.norm(same_grade)/scale),
            "negative_same_grade_spatial_fraction": float(
                np.linalg.norm(opposite_same_grade)/scale),
        },
        "local_beta": {
            "eigenvalues": [float(value) for value in local_values],
            "complex_structure_commutator": float(np.linalg.norm(
                local_beta@complex_structure
                - complex_structure@local_beta)),
            "matrix": local_beta.tolist(),
        },
        "verdict": (
            "LOCAL_NATIVE_BETA_GRADING_DERIVED"
            if np.linalg.norm(offsite)/np.linalg.norm(projector) < 1e-10
            and np.linalg.norm(diagonal-mean_local[None, ...])
                / np.linalg.norm(diagonal) < 1e-10
            and np.linalg.norm(anticommutator)/scale < 1e-10
            else "HODGE_CONFIGURATION_IMAGE_DOES_NOT_DEFINE_LOCAL_BETA"),
        "claim_boundary": (
            "Linear configuration image at the localized rest orbit. A local "
            "beta pass would still require the frequency-amplitude readout, "
            "mass sign, three transport generators and current on the same "
            "phase-space embedding."),
    }
    (HERE/"cube_incidence_hodge_beta_grading_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
