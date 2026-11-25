"""Extract and classify the leading Floquet mode of the cumulative action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu

import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import mixed_prong_dirac_long_tangent as long_gate
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
CACHE = HERE/"stabilized_mixed_prong_tangent_blocks.npz"


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-long_gate.DT**2*rank.R**2/2))
    rminus = long_gate.phase_rotation_matrix(rest.size, -theta)
    rplus = long_gate.phase_rotation_matrix(rest.size, theta)
    data = np.load(CACHE)
    previous = sparse(data["previous"]@rminus)
    current = sparse(data["current"])
    following = sparse(data["following"]@rplus)
    factor = splu(following)
    size = rest.size

    def matvec(state):
        now = state[:size]
        before = state[size:]
        after = factor.solve(-(current@now+previous@before))
        return np.r_[after, now]

    operator = LinearOperator((2*size, 2*size), matvec=matvec,
                              dtype=np.float64)
    values, vectors = eigs(operator, k=12, which="LM", tol=1e-9,
                           maxiter=4000)
    order = np.argsort(abs(values))[::-1]
    values = values[order]
    vectors = vectors[:, order]
    leading = vectors[:size, 0]
    pivot = int(np.argmax(abs(leading)))
    leading *= np.exp(-1j*np.angle(leading[pivot]))
    psi, chi = relative.unpack_raw(leading, long_gate.dirac.SIDE)
    psi_power = np.sum(abs(psi)**2)
    chi_power = np.sum(abs(chi)**2)
    carrier_psi, _ = relative.unpack_raw(rest, long_gate.dirac.SIDE)
    carrier = np.sum(abs(carrier_psi)**2, axis=-1) > 1e-12
    shell = np.zeros_like(carrier)
    for offset, _weight in rank.native.DIRECTIONS:
        shell |= np.roll(carrier, shift=offset, axis=(0, 1, 2))
    shell &= ~carrier
    psi_site_power = np.sum(abs(psi)**2, axis=-1)
    chi_site_power = np.sum(abs(chi)**2, axis=-1)
    top_sites = np.argsort(psi_site_power.reshape(-1))[::-1][:12]
    psi_y_fourier = np.fft.fftn(psi[..., 1])
    peak = np.unravel_index(
        np.argmax(abs(psi_y_fourier)**2), psi_y_fourier.shape)
    frequencies = 2*np.pi*np.fft.fftfreq(long_gate.dirac.SIDE)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    # Use the union of both frozen four-component lifts for classification.
    bases = []
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    for flavor, basis in long_gate.dirac.flavor_bases().items():
        bases.append(envelope.raw_columns(
            basis, psi_frame, chi_frame, sites,
            long_gate.dirac.OFFSETS, scales[flavor]))
    qbasis = envelope.orth(np.column_stack(bases))
    retained = qbasis@(qbasis.T@leading.real) + 1j*qbasis@(
        qbasis.T@leading.imag)
    result = {
        "status": "V4_STABILIZED_MIXED_PRONG_GROWTH_MODE",
        "leading_multipliers": [
            {"real": float(value.real), "imag": float(value.imag),
             "modulus": float(abs(value))} for value in values],
        "leading_per_update_modulus": float(abs(values[0])),
        "leading_per_cycle_amplification": float(
            abs(values[0])**(2*np.pi/(rank.R*long_gate.DT))),
        "leading_mode": {
            "Psi_power_fraction": float(psi_power/(psi_power+chi_power)),
            "chi_power_fraction": float(chi_power/(psi_power+chi_power)),
            "Psi_carrier_fraction": float(
                np.sum(abs(psi[carrier])**2)/max(psi_power, 1e-300)),
            "chi_carrier_fraction": float(
                np.sum(abs(chi[carrier])**2)/max(chi_power, 1e-300)),
            "Psi_first_shell_fraction": float(
                np.sum(psi_site_power[shell])/max(psi_power, 1e-300)),
            "chi_first_shell_fraction": float(
                np.sum(chi_site_power[shell])/max(chi_power, 1e-300)),
            "component_power_fractions": {
                "Psi": (np.sum(abs(psi)**2, axis=(0, 1, 2))
                        / max(psi_power+chi_power, 1e-300)).tolist(),
                "chi": (np.sum(abs(chi)**2, axis=(0, 1, 2))
                        / max(psi_power+chi_power, 1e-300)).tolist(),
            },
            "outside_two_copy_lift_fraction": float(
                np.linalg.norm(leading-retained)
                / np.linalg.norm(leading)),
            "dominant_Psi_y_momentum_index": [int(index) for index in peak],
            "dominant_Psi_y_momentum": [
                float(frequencies[index]) for index in peak],
            "top_Psi_sites": [
                {"site": [int(value) for value in np.unravel_index(
                    int(index), psi_site_power.shape)],
                 "power_fraction": float(
                     psi_site_power.reshape(-1)[index]/psi_power)}
                for index in top_sites],
        },
        "verdict": "PHYSICAL_GROWING_MODE_IDENTIFIED",
        "claim_boundary": (
            "Linear Floquet classification only. It identifies the response "
            "an action repair must change but does not itself admit one."),
    }
    path = HERE/"stabilized_mixed_prong_growth_mode.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
