"""Independent calculus and symmetry checks for the derived kinetic update."""

from __future__ import annotations

import numpy as np

import relative_kinetic_probe as probe


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def kinetic(psi, chi, velocity_psi, velocity_chi, eta):
    relative = (np.cross(chi, velocity_psi)
                - np.cross(velocity_chi, psi))
    return float((np.sum(abs(velocity_psi)**2)
                  + probe.B * np.sum(abs(velocity_chi)**2)
                  + eta / probe.R**2 * np.sum(abs(relative)**2)) / 2)


def main() -> None:
    rng = np.random.default_rng(402)
    shape = (2, 2, 2, 3)
    fields = [rng.normal(size=shape) + 1j * rng.normal(size=shape)
              for _ in range(8)]
    psi, chi, vp, vc, dp, dc, dvp, dvc = fields
    eta = 0.17
    epsilon = 2e-7
    ppsi, pchi, qpsi, qchi = probe.kinetic_derivatives(
        psi - vp / 2, chi - vc / 2, psi + vp / 2, chi + vc / 2, 1.0, eta)
    finite = (kinetic(psi + epsilon * dp, chi + epsilon * dc,
                      vp + epsilon * dvp, vc + epsilon * dvc, eta)
              - kinetic(psi - epsilon * dp, chi - epsilon * dc,
                        vp - epsilon * dvp, vc - epsilon * dvc, eta)) / (2 * epsilon)
    analytic = float(np.real(
        np.vdot(qpsi, dp) + np.vdot(qchi, dc)
        + np.vdot(ppsi, dvp) + np.vdot(pchi, dvc)))
    require(abs(finite - analytic) < 2e-5,
            "kinetic first variation does not match finite difference")

    side, dt = 3, 1 / 256
    previous = rng.normal(size=12 * side**3) * 0.01
    current = rng.normal(size=12 * side**3) * 0.01
    following = rng.normal(size=12 * side**3) * 0.01
    forward = probe.del_residual(previous, current, following, side, dt, eta)
    reverse = probe.del_residual(following, current, previous, side, dt, eta)
    require(np.linalg.norm(forward - reverse) < 1e-9,
            "discrete update is not self-adjoint under endpoint reversal")

    psi0, chi0 = probe.unpack_raw(current, side)
    force = probe.pack_raw(*probe.potential_force(psi0, chi0))
    difference_psi, difference_chi = probe.unpack_raw(
        2 * current - previous - following, side)
    verlet = (probe.pack_raw(difference_psi, probe.B * difference_chi) / dt
              + dt * force)
    zero_eta = probe.del_residual(previous, current, following, side, dt, 0.0)
    require(np.linalg.norm(verlet - zero_eta) < 1e-10,
            "zero-coupling update does not reduce to centered Verlet")
    print("V4_RELATIVE_KINETIC_UPDATE_DERIVATION_PASS")


if __name__ == "__main__":
    main()
