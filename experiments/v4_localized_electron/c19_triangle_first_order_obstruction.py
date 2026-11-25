"""Prove why the symmetric triangle candidate cannot select Dirac's root.

The body/triangle factor has correlation

  C(k)=8 cos(kx) cos(ky) cos(kz).

It is even in momentum and has zero first derivative at the origin.  The
spatial Dirac Hamiltonian 2 alpha.s19(k) is odd and has nonzero derivatives
2 alpha_a at the origin.  The former can converge to the square of the latter,
but no momentum-independent change of four-component basis can turn a zero
first derivative into a nonzero one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c19_triangle_continuum_symbol_audit as triangle
import cube_incidence_extended_packet_gate as dirac


HERE = Path(__file__).resolve().parent


def main() -> None:
    probes = (
        np.asarray((0.173, -0.119, 0.071)),
        np.asarray((0.31, 0.07, -0.23)),
    )
    even_defects = []
    for momentum in probes:
        _, positive = triangle.body_symbol(momentum)
        _, negative = triangle.body_symbol(-momentum)
        even_defects.append(abs(positive - negative))

    rows = []
    zero = np.zeros(3)
    triangle_derivative = triangle.triangle_gradient(zero)
    first_symbol, jacobian = triangle.c19_symbol_and_jacobian(zero)
    for flavor, basis in dirac.flavor_bases().items():
        alpha, _ = dirac.compressed_clifford(basis)
        derivatives = [2.0 * sum(
            jacobian[component, axis] * alpha[component]
            for component in range(3)) for axis in range(3)]
        rows.append({
            "flavor": flavor,
            "s19_at_zero": first_symbol.tolist(),
            "dirac_first_derivative_frobenius_norms": [
                float(np.linalg.norm(value)) for value in derivatives],
        })

    obstructed = bool(
        max(even_defects) < 1e-15
        and np.linalg.norm(triangle_derivative) < 1e-15
        and all(min(row["dirac_first_derivative_frobenius_norms"]) > 1.0
                for row in rows))
    result = {
        "status": "V4_C19_TRIANGLE_FIRST_ORDER_OBSTRUCTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "triangle_symbol_parity": "even",
        "dirac_spatial_symbol_parity": "odd",
        "sample_even_parity_defects": even_defects,
        "triangle_symbol_gradient_at_zero": triangle_derivative.tolist(),
        "copy_rows": rows,
        "verdict": (
            "SYMMETRIC_TRIANGLE_SYMBOL_CANNOT_SELECT_DIRAC_FIRST_ORDER_ROOT"
            if obstructed else "FIRST_ORDER_PARITY_OBSTRUCTION_NOT_PROVED"),
        "root_cause": (
            "The symmetric endpoint/triangle action retains the second-order "
            "Dirac square but discards the sign and component routing carried "
            "by the odd alpha.k term."),
        "corrective_action": (
            "Derive a genuinely oriented C19 face-edge one-form whose full "
            "cubic sum is odd in momentum, reciprocal, and rest-preserving. "
            "It must obtain orientation from live Psi/chi/W geometry, not a "
            "Dirac matrix, corner label, or preferred axis."),
        "claim_boundary": (
            "Scoped obstruction for the present symmetric triangle/body "
            "candidate. It is not a no-go for oriented three-site actions."),
    }
    output = HERE / "c19_triangle_first_order_obstruction.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
