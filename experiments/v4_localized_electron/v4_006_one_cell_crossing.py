"""Test one complete unreset native cell crossing against discrete Dirac.

The initial previous/current layers use the V4-005 packet preparation.  The
native recurrence is then evolved without projection, reload, or target
feedback until the independently evolved centered-C19 Dirac packet first
crosses one lattice cell.  The same frozen V4-003 lift/readout is used at the
endpoint.  Dirac determines only the comparison time and target state.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sparse(value: np.ndarray, tolerance: float = 1e-11) -> csc_matrix:
    result = value.copy()
    result[np.abs(result) < tolerance] = 0.0
    return csc_matrix(result)


def centered_target(initial: np.ndarray, alpha: list[np.ndarray],
                    beta: np.ndarray, step: int, dt: float,
                    mass: float) -> np.ndarray:
    """Exact centered-time negative-frequency packet at an integer step."""
    side = initial.shape[0]
    fourier = np.fft.fftn(initial, axes=(0, 1, 2))
    result = np.empty_like(fourier)
    frequencies = 2*np.pi*np.fft.fftfreq(side)
    theta0 = float(np.arccos(1-dt**2*mass**2/2))
    for index in itertools.product(range(side), repeat=3):
        momentum = np.asarray([frequencies[value] for value in index])
        import cube_incidence_extended_packet_gate as dirac
        transport = dirac.c19_first_symbol(momentum, 1.0)
        hamiltonian = mass*beta
        for axis in range(3):
            hamiltonian = hamiltonian+transport[axis]*alpha[axis]
        energy = float(np.sqrt(mass**2+transport@transport))
        theta = float(np.arccos(np.clip(
            1-dt**2*energy**2/2, -1.0, 1.0)))
        laboratory = (
            np.cos(step*theta)*np.eye(4)
            - 1j*np.sin(step*theta)*hamiltonian/energy)
        result[index] = (
            np.exp(-1j*step*theta0)*laboratory@fourier[index])
    return np.fft.ifftn(result, axes=(0, 1, 2))


def remove_growing_modes(current: np.ndarray, previous: np.ndarray,
                         factor, middle_operator, previous_operator,
                         count: int = 24,
                         tolerance: float = 1e-8):
    """Project the two committed layers onto the tangent center-stable space.

    This is a one-time native preparation correction.  Right and left
    Floquet vectors are derived solely from the declared action recurrence;
    no Dirac target enters the projection.  The subsequent evolution remains
    unrestricted.
    """
    size = current.size

    def forward(state):
        now, before = state[:size], state[size:]
        after = factor.solve(-(
            middle_operator@now + previous_operator@before))
        return np.r_[after, now]

    def transpose(state):
        upper, lower = state[:size], state[size:]
        solved = factor.solve(upper, trans="T")
        return np.r_[
            -(middle_operator.T@solved)+lower,
            -(previous_operator.T@solved)]

    shape = (2*size, 2*size)
    operator = LinearOperator(shape, matvec=forward, rmatvec=transpose,
                              dtype=np.float64)
    adjoint = LinearOperator(shape, matvec=transpose, rmatvec=forward,
                             dtype=np.float64)
    values, right = eigs(operator, k=count, which="LM", tol=1e-10,
                         maxiter=10000)
    left_values, left_all = eigs(adjoint, k=count, which="LM", tol=1e-10,
                                 maxiter=10000)
    selected = np.flatnonzero(np.abs(values) > 1+tolerance)
    state = np.r_[current, previous].astype(np.complex128)
    if selected.size == 0:
        return current, previous, {
            "removed_mode_count": 0,
            "leading_multiplier_modulus": float(np.max(np.abs(values))),
            "relative_preparation_change": 0.0,
        }
    right_u = right[:, selected]
    left_columns = []
    for value in values[selected]:
        index = int(np.argmin(np.abs(left_values-np.conj(value))))
        left_columns.append(left_all[:, index])
    left_u = np.column_stack(left_columns)
    pairing = left_u.conj().T@right_u
    coefficients = np.linalg.solve(pairing, left_u.conj().T@state)
    corrected = state-right_u@coefficients
    imaginary_fraction = float(
        np.linalg.norm(corrected.imag)/max(np.linalg.norm(corrected), 1e-30))
    corrected = corrected.real
    diagnostics = {
        "removed_mode_count": int(selected.size),
        "removed_multipliers": [
            {"real": float(values[index].real),
             "imag": float(values[index].imag),
             "modulus": float(abs(values[index]))}
            for index in selected],
        "leading_multiplier_modulus": float(np.max(np.abs(values))),
        "relative_preparation_change": float(
            np.linalg.norm(corrected-np.r_[current, previous])
            / max(np.linalg.norm(np.r_[current, previous]), 1e-30)),
        "postprojection_imaginary_fraction": imaginary_fraction,
        "biorthogonal_pairing_condition": float(np.linalg.cond(pairing)),
    }
    return corrected[:size], corrected[size:], diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=1/256)
    parser.add_argument("--copy", choices=("plus", "minus"), default="plus")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--wave-number", type=float, default=0.35)
    parser.add_argument("--packet-width", type=float, default=1.35)
    parser.add_argument("--chi-scale-real", type=float)
    parser.add_argument("--chi-scale-imag", type=float, default=0.0)
    parser.add_argument("--scan-step", type=int, default=64)
    parser.add_argument("--maximum-steps", type=int, default=60000)
    parser.add_argument("--remove-growing-modes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tangent_cache = args.tangent_cache.resolve()
    rest_cache = args.rest_cache.resolve()
    sys.path.insert(0, str(HERE))
    import personal_touch_unit_geometric_canonical as canonical
    canonical.CACHE = rest_cache
    import canonical_noncoplanar_bifurcation_audit as native_time
    import canonical_transfer_oscillator_spectrum as native_spectrum
    import cube_incidence_extended_packet_dirac_current as current
    import cube_incidence_extended_packet_gate as dirac
    import current_action_localized_envelope_closure as envelope
    from v4_005_short_time_transport import readout_solver

    cache = np.load(tangent_cache)
    previous_block, middle_block, following_block = (
        cache[name] for name in ("previous", "current", "following"))
    rest = np.load(rest_cache)["rest"]
    dimension = rest.size
    theta = float(np.arccos(
        1-args.dt**2*native_time.CHI0**2/2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    factor = splu(sparse(following_block@rotate_plus))
    middle_operator = sparse(middle_block)
    previous_operator = sparse(previous_block@rotate_minus)

    basis = dirac.flavor_bases()[args.copy]
    alpha, beta = dirac.compressed_clifford(basis)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    chi_scale = (
        complex(1/np.sqrt(native_time.B), 0.0)
        if args.chi_scale_real is None else
        complex(args.chi_scale_real, args.chi_scale_imag))
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, offsets, chi_scale)
    solve_readout, ridge, condition = readout_solver(raw)
    packet_previous, packet_initial, _, momentum = dirac.packet(
        args.axis, args.sign, alpha, beta, args.dt,
        width=args.packet_width,
        carrier_wave_number=args.wave_number,
        centered_time=True, frequency_branch="negative",
        transport_scale=1.0)
    initial_center = dirac.circular_center(packet_initial)

    first_target = centered_target(
        packet_initial, alpha, beta, 1, args.dt,
        dirac.COMPARATOR_MASS)
    first_shift = dirac.wrapped_shift(
        initial_center, dirac.circular_center(first_target))[args.axis]
    direction = int(np.sign(first_shift))
    if direction == 0:
        raise RuntimeError("Dirac packet has zero initial longitudinal shift")

    bracket = None
    prior_step = 0
    prior_shift = 0.0
    for step in range(args.scan_step, args.maximum_steps+1,
                      args.scan_step):
        target = centered_target(
            packet_initial, alpha, beta, step, args.dt,
            dirac.COMPARATOR_MASS)
        shift = float(dirac.wrapped_shift(
            initial_center, dirac.circular_center(target))[args.axis])
        if direction*shift >= 1.0:
            bracket = (prior_step, step)
            break
        prior_step, prior_shift = step, shift
    if bracket is None:
        raise RuntimeError(
            "independent Dirac packet did not cross one cell within scan")
    lower, upper = bracket
    while upper-lower > 1:
        step = (lower+upper)//2
        target = centered_target(
            packet_initial, alpha, beta, step, args.dt,
            dirac.COMPARATOR_MASS)
        shift = float(dirac.wrapped_shift(
            initial_center, dirac.circular_center(target))[args.axis])
        if direction*shift >= 1.0:
            upper = step
        else:
            lower = step
    crossing_step = upper
    target = centered_target(
        packet_initial, alpha, beta, crossing_step, args.dt,
        dirac.COMPARATOR_MASS)

    q_previous = raw@dirac.real_coordinates(packet_previous)
    q_current = raw@dirac.real_coordinates(packet_initial)
    stable_preparation = None
    if args.remove_growing_modes:
        q_current, q_previous, stable_preparation = remove_growing_modes(
            q_current, q_previous, factor, middle_operator,
            previous_operator)
    sample_steps = set(np.linspace(
        0, crossing_step, 21, dtype=int).tolist())
    samples = []
    for step in range(crossing_step+1):
        if step in sample_steps:
            target_now = centered_target(
                packet_initial, alpha, beta, step, args.dt,
                dirac.COMPARATOR_MASS)
            recovered = solve_readout(q_current)
            native_field = (
                recovered.reshape(-1, 2)@np.asarray((1.0, 1.0j))
            ).reshape(5, 5, 5, 4)
            samples.append({
                "step": step,
                "time": step*args.dt,
                "native_center_shift": dirac.wrapped_shift(
                    initial_center,
                    dirac.circular_center(native_field)).tolist(),
                "dirac_center_shift": dirac.wrapped_shift(
                    initial_center,
                    dirac.circular_center(target_now)).tolist(),
                "outside_fixed_readout_fraction": float(
                    np.linalg.norm(q_current-raw@recovered)
                    / max(np.linalg.norm(q_current), 1e-30)),
            })
        if step == crossing_step:
            break
        q_following = factor.solve(-(
            middle_operator@q_current+previous_operator@q_previous))
        q_previous, q_current = q_current, q_following

    recovered = solve_readout(q_current)
    native_field = (
        recovered.reshape(-1, 2)@np.asarray((1.0, 1.0j))
    ).reshape(5, 5, 5, 4)
    target_real = dirac.real_coordinates(target)
    target_i = dirac.real_coordinates(1j*target)
    phase_basis = np.column_stack((target_real, target_i))
    coefficients, *_ = np.linalg.lstsq(
        phase_basis, recovered, rcond=1e-12)
    aligned = phase_basis@(
        coefficients/max(np.linalg.norm(coefficients), 1e-30))
    density_native, current_native = current.observables(native_field, alpha)
    density_target, current_target = current.observables(target, alpha)
    native_shift = dirac.wrapped_shift(
        initial_center, dirac.circular_center(native_field))
    target_shift = dirac.wrapped_shift(
        initial_center, dirac.circular_center(target))
    result = {
        "stage": "V4-006",
        "status": "ONE_CELL_CROSSING_AUDIT",
        "action_sha256": sha256(HERE/"live_clifford_bond_action.py"),
        "tangent_cache_sha256": sha256(tangent_cache),
        "rest_cache_sha256": sha256(rest_cache),
        "dt": args.dt,
        "copy": args.copy,
        "axis": "xyz"[args.axis],
        "input_sign": args.sign,
        "observed_Dirac_direction": direction,
        "momentum": momentum.tolist(),
        "chi_scale": {"real": float(chi_scale.real),
                      "imag": float(chi_scale.imag)},
        "crossing_step": crossing_step,
        "crossing_time": crossing_step*args.dt,
        "readout_sha256": hashlib.sha256(
            np.ascontiguousarray(raw).tobytes()).hexdigest(),
        "readout_ridge": ridge,
        "readout_condition": condition,
        "Dirac_center_shift": target_shift.tolist(),
        "native_center_shift": native_shift.tolist(),
        "longitudinal_center_error": float(abs(
            native_shift[args.axis]-target_shift[args.axis])),
        "phase_aligned_readout_error": float(
            np.linalg.norm(recovered-aligned)
            / max(np.linalg.norm(target_real), 1e-30)),
        "outside_fixed_readout_fraction": float(
            np.linalg.norm(q_current-raw@recovered)
            / max(np.linalg.norm(q_current), 1e-30)),
        "density_relative_error": float(
            abs(density_native-density_target)/max(density_target, 1e-30)),
        "current_relative_error": float(
            np.linalg.norm(current_native-current_target)
            / max(np.linalg.norm(current_target), 1e-30)),
        "native_current": current_native.tolist(),
        "Dirac_current": current_target.tolist(),
        "longitudinal_current_sign_matches": bool(
            np.sign(current_native[args.axis])
            == np.sign(current_target[args.axis])),
        "samples": samples,
        "unreset_native_evolution": True,
        "receiver_preloaded_or_reset": False,
        "one_time_action_stable_preparation": stable_preparation,
        "Dirac_used_in_native_update": False,
        "claim_boundary": (
            "One complete cell crossing of one signed axial packet in the "
            "unrestricted frozen-action tangent evolution. V4-007 must "
            "establish an unreset second crossing in both directions."),
    }
    output = args.output if args.output.is_absolute() else HERE/args.output
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
