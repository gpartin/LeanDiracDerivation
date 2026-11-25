"""Screen native scalar orientations of the common Psi--chi pair connection.

All candidates use the same reciprocal one-form.  They differ only in the
phase-invariant local scalar whose bond contrast supplies the orientation.
Directional derivatives are used first; no full action matrix or coefficient
fit is built for a channel that fails this screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import localized_stability as native
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
FEATURES = (
    "rho_Psi",
    "rho_chi",
    "area_density",
    "sigma_Psi_anisotropy",
    "sigma_chi_anisotropy",
    "area_bilinear_anisotropy",
)


def local_scalar(raw: torch.Tensor, name: str) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    rho_psi = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1))
    rho_chi = torch.real(torch.sum(torch.conj(chi)*chi, dim=-1))
    area = torch.linalg.cross(chi, psi, dim=-1)
    if name == "rho_Psi":
        return rho_psi/native_time.CHI0**2
    if name == "rho_chi":
        return rho_chi/native_time.CHI0**2
    if name == "area_density":
        return torch.real(torch.sum(torch.conj(area)*area, dim=-1))/native_time.CHI0**4
    if name == "sigma_Psi_anisotropy":
        sigma = torch.sum(psi*psi, dim=-1)
        return torch.real(torch.conj(sigma)*sigma)/native_time.CHI0**4
    if name == "sigma_chi_anisotropy":
        sigma = torch.sum(chi*chi, dim=-1)
        return torch.real(torch.conj(sigma)*sigma)/native_time.CHI0**4
    if name == "area_bilinear_anisotropy":
        sigma = torch.sum(area*area, dim=-1)
        return torch.real(torch.conj(sigma)*sigma)/native_time.CHI0**8
    raise ValueError(name)


def pair(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    return chi[..., :, None]*psi[..., None, :]


def one_form(left: torch.Tensor, right: torch.Tensor,
             feature: str) -> torch.Tensor:
    midpoint = (left+right)/2
    scalar = local_scalar(midpoint, feature)
    z = pair(midpoint)
    dz = pair(right)-pair(left)
    result = left.new_zeros((), dtype=torch.float64)
    for offset, weight in native.DIRECTIONS:
        scalar_j = torch.roll(scalar, shifts=offset, dims=(0, 1, 2))
        z_j = torch.roll(z, shifts=offset, dims=(0, 1, 2))
        dz_j = torch.roll(dz, shifts=offset, dims=(0, 1, 2))
        pairing = torch.sum(
            torch.conj(z)*dz_j-torch.conj(z_j)*dz, dim=(-2, -1))
        result = result+weight*torch.sum(
            (scalar-scalar_j)*torch.imag(pairing))
    return result/native_time.CHI0**4


def directional_blocks(rest: np.ndarray, feature: str,
                       vectors: dict[str, np.ndarray]):
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]

    def residual(left, middle, right):
        action = one_form(left, middle, feature)+one_form(middle, right, feature)
        return torch.autograd.grad(action, middle, create_graph=True)[0]

    action = one_form(layers[0], layers[1], feature)+one_form(
        layers[1], layers[2], feature)
    rest_force = torch.autograd.grad(
        action, layers[1], retain_graph=True)[0].detach().numpy()
    products = {}
    for label, vector in vectors.items():
        pieces = []
        for argument, physical_vector in enumerate((
                rotation.T@vector, vector, rotation@vector)):
            def function(value, selected=argument):
                trial = layers.copy()
                trial[selected] = value
                return residual(*trial)
            _value, tangent = torch.autograd.functional.jvp(
                function, layers[argument],
                torch.tensor(physical_vector, dtype=torch.float64),
                create_graph=False, strict=True)
            pieces.append(tangent.detach().numpy())
        minus, zero, plus = pieces
        products[label] = {
            "mass": -0.5*dt*(plus+minus),
            "gyro": -(plus-minus),
            "stiffness": -(plus+zero+minus)/dt,
        }
    return float(action.detach()), rest_force, products


def main() -> None:
    bare = np.load(canonical.CACHE)
    residual_cache = np.load(
        HERE/"density_connection_fastest_residual_mode.npz")
    mode = residual_cache["current"]
    psi, chi = native.unpack(bare["rest"], SIDE)
    translations = native.translation_vectors(psi, chi)
    vectors = {"mode_real": mode.real, "mode_imag": mode.imag}
    vectors.update({f"translation_{axis}": translations[:, axis]
                    for axis in range(3)})
    rows = []
    response_columns = []
    factor = lu_factor(
        residual_cache["mass"]/native_time.DT+residual_cache["gyro"]/2)
    for feature in FEATURES:
        action, force, products = directional_blocks(
            bare["rest"], feature, vectors)
        mode_mass = sum(np.dot(vectors[name], products[name]["mass"])
                        for name in ("mode_real", "mode_imag"))
        mode_gyro = np.sqrt(sum(
            np.linalg.norm(products[name]["gyro"])**2
            for name in ("mode_real", "mode_imag")))
        mode_stiffness = sum(
            np.dot(vectors[name], products[name]["stiffness"])
            for name in ("mode_real", "mode_imag"))
        transport = []
        concatenated = []
        for axis in range(3):
            label = f"translation_{axis}"
            product = products[label]
            following = -lu_solve(factor, (
                native_time.DT*product["stiffness"]
                + product["gyro"]
                + product["mass"]@np.zeros_like(translations[:, axis])))
            transport.append(float(np.linalg.norm(following)))
            concatenated.append(following)
        response_columns.append(np.concatenate(concatenated))
        rows.append({
            "feature": feature,
            "two_interval_action": action,
            "rest_DEL_norm": float(np.linalg.norm(force)),
            "mode_mass_directional_sum": float(mode_mass),
            "mode_gyro_response_norm": float(mode_gyro),
            "mode_stiffness_directional_sum": float(mode_stiffness),
            "translation_response_norms": transport,
        })
    matrix = np.column_stack(response_columns)
    singular = np.linalg.svd(matrix, compute_uv=False)
    result = {
        "status": "V4_DENSITY_PAIR_CONNECTION_BASIS_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "channels": rows,
        "three_axis_response_singular_values": singular.tolist(),
        "three_axis_response_rank": int(np.sum(
            singular > 1e-10*max(float(singular[0]), 1.0))),
        "verdict": "SCALAR_ORIENTATION_BASIS_SCREENED",
        "claim_boundary": (
            "Directional derivatives identify independent action responses. "
            "They do not replace a full matrix spectrum or admit a term."),
    }
    (HERE/"density_pair_connection_basis_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
