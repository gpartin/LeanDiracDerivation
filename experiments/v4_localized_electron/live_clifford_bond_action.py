"""Research action using the Clifford connection derived by live Psi links.

For each positive face bond, the normalized phase of ``Psi_i^dagger Psi_j``
is the native link variable.  The three-prong fields are exchanged through
the real antisymmetric spacetime one-form

  Re[(D_t Psi_i)^dagger u_ij Psi_j
     -(D_t Psi_j)^dagger conj(u_ij) Psi_i]

and the corresponding chi payload.  ``D_t Psi`` removes only the exact
centered rest clock, so the term and its force are tested to vanish on the
rest orbit while remaining active on spatial motion.  The link phases—not a
fixed alpha matrix—derive the anticommuting cube connection.

This module is an unadmitted action candidate.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import density_mold_connection_probe as local
import wake_stabilized_completed_action as incumbent
import localized_stability as c19
import overlap_material_action as overlap_material
import triangle_movement_action as triangle


SIDE = 5
KAPPA = 1.0/63.0
SCALE = float(os.environ.get("LFM_LIVE_CLIFFORD_SCALE", "1"))
PLAQUETTE_SCALE = float(os.environ.get("LFM_LIVE_PLAQUETTE_SCALE", "0"))
SPACETIME_PLAQUETTE_SCALE = float(
    os.environ.get("LFM_LIVE_SPACETIME_PLAQUETTE_SCALE", "0"))
SYMPLECTIC_PLAQUETTE_SCALE = float(
    os.environ.get("LFM_LIVE_SYMPLECTIC_PLAQUETTE_SCALE", "0"))
PARALLEL_OVERLAP_SCALE = float(
    os.environ.get("LFM_LIVE_PARALLEL_OVERLAP_SCALE", "0"))
PARALLEL_FACE_SCALE = float(
    os.environ.get("LFM_LIVE_PARALLEL_FACE_SCALE", "0"))
PARALLEL_MATCHED_SCALE = float(
    os.environ.get("LFM_LIVE_PARALLEL_MATCHED_SCALE", "0"))
PARALLEL_MATCHED_POTENTIAL_SCALE = float(
    os.environ.get("LFM_LIVE_PARALLEL_MATCHED_POTENTIAL_SCALE", "1"))
TRIANGLE_DELTA_SCALE = float(
    os.environ.get("LFM_LIVE_TRIANGLE_DELTA_SCALE", "0"))
MATERIAL_AREA_SCALE = float(
    os.environ.get("LFM_LIVE_MATERIAL_AREA_SCALE", "0"))
CHI_PAYLOAD = float(os.environ.get("LFM_LIVE_CLIFFORD_CHI_PAYLOAD", "1"))
LINK_SOURCE = os.environ.get("LFM_LIVE_CLIFFORD_LINK", "psi_phase")


def norm2(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value)*value, dim=-1))


def smooth_wedge_gate(field: torch.Tensor,
                      neighbor: torch.Tensor) -> torch.Tensor:
    """Smooth normalized Gram area, fixed to one on orthogonal vacua.

    The earlier diagnostic divided by ``rho_i rho_j`` and selected a zero
    branch when either norm vanished.  That is adequate on the nonzero test
    orbit but does not define a globally smooth action.  Native ``kappa``
    supplies the only already-declared positive regulator.  The prefactor
    keeps the gate exactly one when both fields have norm ``chi0`` and are
    orthogonal, so this is a regularization rather than a retuning.
    """
    wedge = torch.linalg.cross(field, neighbor, dim=-1)
    regulator = KAPPA*native_time.CHI0**2
    normalization = (1.0+KAPPA)**2
    return (normalization*norm2(wedge)
            / ((norm2(field)+regulator)
               * (norm2(neighbor)+regulator)))


def link_phase(field: torch.Tensor, neighbor: torch.Tensor,
               mode: str) -> torch.Tensor:
    overlap = torch.sum(torch.conj(field)*neighbor, dim=-1)
    if mode == "raw":
        return overlap/native_time.CHI0**2
    overlap_magnitude2 = torch.real(torch.conj(overlap)*overlap)
    # This must be the phase of the overlap, not its normalized magnitude.
    # Dividing by ||field||||neighbor|| leaves the ray-overlap cosine in the
    # transporter and does not reproduce the live Clifford generators proved
    # by ``live_clifford_link_geometry``.
    denominator = torch.sqrt(torch.clamp(overlap_magnitude2, min=1e-28))
    # The action is inactive when the native edge overlap vanishes.  The zero
    # branch is not asserted to be a Frechet derivative at that singular
    # point; directional vacuum controls remain mandatory.
    result = torch.where(overlap_magnitude2 > 1e-28,
                         overlap/denominator,
                         torch.zeros_like(overlap))
    if mode in ("conjugate_phase", "frozen_conjugate_phase"):
        result = torch.conj(result)
    if mode in ("frozen_phase", "frozen_conjugate_phase"):
        # Diagnostic control only.  Detaching a native link is not an
        # admissible reciprocal action, but distinguishes the desired
        # connection from the reciprocal response of its composite source.
        result = result.detach()
    return result


def ray_transporter(local_field: torch.Tensor,
                    neighbor_field: torch.Tensor,
                    link: torch.Tensor, frozen: bool) -> torch.Tensor:
    """Map the neighbor ray into the local ray with the live link phase."""
    local_norm = torch.sqrt(torch.clamp(norm2(local_field), min=1e-28))
    neighbor_norm = torch.sqrt(torch.clamp(norm2(neighbor_field), min=1e-28))
    outer = (local_field[..., :, None]
             * torch.conj(neighbor_field[..., None, :]))
    outer = outer/(local_norm*neighbor_norm)[..., None, None]
    active = ((norm2(local_field)*norm2(neighbor_field)) > 1e-28)
    outer = torch.where(active[..., None, None], outer,
                        torch.zeros_like(outer))
    result = link[..., None, None]*outer
    return result.detach() if frozen else result


def apply(matrix: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ab,...b->...a", matrix, value)


def one_form(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    side = round((left.numel()/12)**(1/3))
    if side**3*12 != left.numel():
        raise ValueError("native state is not a cubic Psi/chi lattice")
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    centered_clock = 2*np.tan(theta/2)/dt
    psi_covariant_dot = psi_dot-1j*centered_clock*psi

    result = left.new_zeros((), dtype=torch.float64)
    if LINK_SOURCE == "joint_c19_wedge_metric_sq":
        # The directed C19 list contains both orientations.  weight/(2*24)
        # removes that double count and normalizes by the C19 weight sum.
        bonds = [(tuple(offset), weight/48.0)
                 for offset, weight in c19.DIRECTIONS]
    else:
        bonds = []
        for axis in range(3):
            shifts = [0, 0, 0]
            shifts[axis] = -1
            bonds.append((tuple(shifts), 1.0))
    for shifts, bond_factor in bonds:
        neighbor_psi = torch.roll(psi, shifts=shifts, dims=(0, 1, 2))
        neighbor_chi = torch.roll(chi, shifts=shifts, dims=(0, 1, 2))
        neighbor_psi_dot = torch.roll(
            psi_covariant_dot, shifts=tuple(shifts), dims=(0, 1, 2))
        neighbor_chi_dot = torch.roll(
            chi_dot, shifts=tuple(shifts), dims=(0, 1, 2))
        if LINK_SOURCE in (
                "chi_wedge_metric", "chi_wedge_metric_sq",
                "joint_wedge_metric_sq", "joint_c19_wedge_metric_sq",
                "joint_face_wedge_metric_sq_smooth",
                "joint_face_wedge_laplacian_smooth",
                "joint_face_laplacian_plus_psi_frame_phase",
                "joint_face_laplacian_plus_parity_pair",
                "joint_face_laplacian_plus_area_oscillator",
                "joint_face_laplacian_plus_frame_oscillator",
                "joint_face_laplacian_plus_psi_area_oscillator",
                "joint_face_laplacian_plus_psi_area_oscillator_frame_phase",
                "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy",
                "joint_face_laplacian_plus_triangle_holonomy"):
            # Symmetric, phase-neutral kinetic communication.  The normalized
            # chi wedge is zero in uniform exterior substrate and maximal at
            # the orthogonal carrier/exterior boundary.  Direct C3 departure
            # velocities are coupled component-for-component; full variation
            # supplies the reciprocal chi response through the live gate.
            chi_wedge = torch.linalg.cross(chi, neighbor_chi, dim=-1)
            wedge_norm2 = norm2(chi_wedge)
            chi_product = norm2(chi)*norm2(neighbor_chi)
            denominator = torch.sqrt(torch.clamp(chi_product, min=1e-28))
            if LINK_SOURCE in (
                    "chi_wedge_metric_sq", "joint_wedge_metric_sq",
                    "joint_c19_wedge_metric_sq"):
                # The squared normalized wedge has the same unit response at
                # an orthogonal carrier/exterior interface, while avoiding
                # the |wedge| cusp on aligned exterior bonds.  It is the
                # smooth, phase-neutral Gram-area invariant needed by the
                # implicit variational update.
                wedge_gate = torch.where(
                    chi_product > 1e-28,
                    wedge_norm2/torch.clamp(chi_product, min=1e-28),
                    torch.zeros_like(chi_product))
            elif LINK_SOURCE in (
                    "joint_face_wedge_metric_sq_smooth",
                    "joint_face_wedge_laplacian_smooth",
                    "joint_face_laplacian_plus_psi_frame_phase",
                    "joint_face_laplacian_plus_parity_pair",
                    "joint_face_laplacian_plus_area_oscillator",
                    "joint_face_laplacian_plus_frame_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator_frame_phase",
                    "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy",
                    "joint_face_laplacian_plus_triangle_holonomy"):
                wedge_gate = smooth_wedge_gate(chi, neighbor_chi)
            else:
                wedge_gate = torch.where(
                    chi_product > 1e-28,
                    torch.sqrt(torch.clamp(wedge_norm2, min=1e-28))/denominator,
                    torch.zeros_like(chi_product))
            if LINK_SOURCE in (
                    "joint_face_wedge_laplacian_smooth",
                    "joint_face_laplacian_plus_psi_frame_phase",
                    "joint_face_laplacian_plus_parity_pair",
                    "joint_face_laplacian_plus_area_oscillator",
                    "joint_face_laplacian_plus_frame_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator_frame_phase",
                    "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy",
                    "joint_face_laplacian_plus_triangle_holonomy"):
                # Positive graph-Laplacian completion.  Only relative motion
                # of neighboring prongs enters; the common covariant rest
                # clock is an exact null direction.  Varying an undirected
                # face sum supplies the six incident off-diagonal handoff
                # terms and their required 6/chi0 diagonal completion.
                psi_difference = psi_covariant_dot-neighbor_psi_dot
                kinetic_pair = 0.5*wedge_gate*norm2(psi_difference)
            else:
                kinetic_pair = torch.sum(
                    torch.conj(psi_covariant_dot)
                    * (wedge_gate[..., None]*neighbor_psi_dot), dim=-1)
            if LINK_SOURCE in (
                    "joint_wedge_metric_sq", "joint_c19_wedge_metric_sq",
                    "joint_face_wedge_metric_sq_smooth"):
                # Complete native prong-to-prong communication.  Chi carries
                # its already-declared inertia B, so the same dimensionless
                # geometric coefficient modifies both canonical kinetic
                # metrics without introducing a second fitted parameter.
                kinetic_pair = kinetic_pair + native_time.B*torch.sum(
                    torch.conj(chi_dot)
                    * (wedge_gate[..., None]*neighbor_chi_dot), dim=-1)
            elif LINK_SOURCE in (
                    "joint_face_wedge_laplacian_smooth",
                    "joint_face_laplacian_plus_psi_frame_phase",
                    "joint_face_laplacian_plus_parity_pair",
                    "joint_face_laplacian_plus_area_oscillator",
                    "joint_face_laplacian_plus_frame_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator",
                    "joint_face_laplacian_plus_psi_area_oscillator_frame_phase",
                    "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy",
                    "joint_face_laplacian_plus_triangle_holonomy"):
                chi_difference = chi_dot-neighbor_chi_dot
                kinetic_pair = (kinetic_pair
                                + 0.5*native_time.B*wedge_gate
                                * norm2(chi_difference))
            result = result+bond_factor*torch.sum(torch.real(kinetic_pair))
            if LINK_SOURCE in (
                    "joint_face_laplacian_plus_psi_frame_phase",
                    "joint_face_laplacian_plus_parity_pair",
                    "joint_face_laplacian_plus_psi_area_oscillator_frame_phase"):
                # Odd, signed companion to the even face metric.  The link
                # phase and rank-one ray transporter are derived from the
                # live Psi endpoints.  Endpoint exchange supplies the
                # reciprocal reverse map; no direction, copy, alpha matrix,
                # or carrier label is selected.
                modes = (("phase",) if LINK_SOURCE in (
                    "joint_face_laplacian_plus_psi_frame_phase",
                    "joint_face_laplacian_plus_psi_area_oscillator_frame_phase")
                    else ("phase", "conjugate_phase"))
                parity_weight = 1.0/len(modes)
                for mode in modes:
                    live_link = link_phase(psi, neighbor_psi, mode)
                    psi_forward = ray_transporter(
                        psi, neighbor_psi, live_link, False)
                    psi_reverse = torch.conj(psi_forward).transpose(-1, -2)
                    chi_forward = ray_transporter(
                        chi, neighbor_chi, live_link, False)
                    chi_reverse = torch.conj(chi_forward).transpose(-1, -2)
                    psi_pair = torch.sum(
                        torch.conj(psi_covariant_dot)
                        * apply(psi_forward, neighbor_psi)
                        - torch.conj(neighbor_psi_dot)
                        * apply(psi_reverse, psi), dim=-1)
                    chi_pair = torch.sum(
                        torch.conj(chi_dot)*apply(chi_forward, neighbor_chi)
                        - torch.conj(neighbor_chi_dot)
                        * apply(chi_reverse, chi), dim=-1)
                    result = result+parity_weight*bond_factor*torch.sum(
                        torch.real(psi_pair+CHI_PAYLOAD*chi_pair))
            continue
        if LINK_SOURCE == "chi_triple_direct":
            # A complete signed interface phase survives even when the two
            # chi vectors are orthogonal.  The scalar triple product is cubic
            # covariant, reverses under endpoint exchange, and carries the
            # source Psi orientation needed to distinguish all eight corners.
            chi_wedge = torch.linalg.cross(chi, neighbor_chi, dim=-1)
            triple = torch.sum((psi+neighbor_psi)*chi_wedge, dim=-1)
            triple_magnitude2 = torch.real(torch.conj(triple)*triple)
            denominator = torch.sqrt(torch.clamp(
                triple_magnitude2, min=1e-28))
            signed_link = torch.where(
                triple_magnitude2 > 1e-28,
                triple/denominator,
                torch.zeros_like(triple))
            psi_neighbor_payload = signed_link[..., None]*neighbor_psi
            # signed_link is antisymmetric under endpoint exchange, so the
            # same link (not its conjugate) multiplies the reverse payload.
            psi_local_payload = signed_link[..., None]*psi
            psi_pair = torch.sum(
                torch.conj(psi_covariant_dot)*psi_neighbor_payload
                - torch.conj(neighbor_psi_dot)*psi_local_payload, dim=-1)
            result = result+bond_factor*torch.sum(torch.real(psi_pair))
            continue
        if LINK_SOURCE == "chi_wedge_direct":
            # The carrier chi and ordinary exterior chi are orthogonal, so a
            # scalar overlap link vanishes precisely at the handoff boundary.
            # Their normalized wedge instead gives the sine of their relative
            # orientation and remains nonzero there.  It gates direct C3
            # component exchange without selecting a prepared direction.
            chi_wedge = torch.linalg.cross(chi, neighbor_chi, dim=-1)
            wedge_norm2 = norm2(chi_wedge)
            chi_product = norm2(chi)*norm2(neighbor_chi)
            denominator = torch.sqrt(torch.clamp(chi_product, min=1e-28))
            wedge_gate = torch.where(
                chi_product > 1e-28,
                torch.sqrt(torch.clamp(wedge_norm2, min=1e-28))/denominator,
                torch.zeros_like(chi_product))
            psi_neighbor_payload = wedge_gate[..., None]*neighbor_psi
            psi_local_payload = wedge_gate[..., None]*psi
            psi_pair = torch.sum(
                torch.conj(psi_covariant_dot)*psi_neighbor_payload
                - torch.conj(neighbor_psi_dot)*psi_local_payload, dim=-1)
            # Chi participates reciprocally through the variation of the live
            # wedge gate.  No independent chi velocity payload is needed.
            result = result+bond_factor*torch.sum(torch.real(psi_pair))
            continue
        frame_mode = LINK_SOURCE in (
            "psi_frame_phase", "psi_frozen_frame_phase",
            "chi_frame_phase", "chi_frozen_frame_phase",
            "psi_frame_conjugate_phase",
            "psi_frozen_frame_conjugate_phase",
            "chi_frame_conjugate_phase",
            "chi_frozen_frame_conjugate_phase")
        if LINK_SOURCE.startswith("chi_"):
            source, neighbor_source = chi, neighbor_chi
            mode = LINK_SOURCE.removeprefix("chi_")
        else:
            source, neighbor_source = psi, neighbor_psi
            mode = LINK_SOURCE.removeprefix("psi_")
        frozen_frame = mode in (
            "frozen_frame_phase", "frozen_frame_conjugate_phase")
        if frame_mode:
            mode = ("conjugate_phase" if "conjugate" in mode else "phase")
        link = link_phase(source, neighbor_source, mode)
        if frame_mode:
            psi_forward = ray_transporter(
                psi, neighbor_psi, link, frozen_frame)
            psi_reverse = torch.conj(psi_forward).transpose(-1, -2)
            chi_forward = ray_transporter(
                chi, neighbor_chi, link, frozen_frame)
            chi_reverse = torch.conj(chi_forward).transpose(-1, -2)
            psi_neighbor_payload = apply(psi_forward, neighbor_psi)
            psi_local_payload = apply(psi_reverse, psi)
            chi_neighbor_payload = apply(chi_forward, neighbor_chi)
            chi_local_payload = apply(chi_reverse, chi)
        else:
            psi_neighbor_payload = link[..., None]*neighbor_psi
            psi_local_payload = torch.conj(link)[..., None]*psi
            chi_neighbor_payload = link[..., None]*neighbor_chi
            chi_local_payload = torch.conj(link)[..., None]*chi
        psi_pair = torch.sum(
            torch.conj(psi_covariant_dot)*psi_neighbor_payload
            - torch.conj(neighbor_psi_dot)*psi_local_payload, dim=-1)
        chi_pair = torch.sum(
            torch.conj(chi_dot)*chi_neighbor_payload
            - torch.conj(neighbor_chi_dot)*chi_local_payload, dim=-1)
        # Face relative weight two, divided by two because every undirected
        # face bond is represented by its two antisymmetric endpoints.
        result = result+bond_factor*torch.sum(
            torch.real(psi_pair+CHI_PAYLOAD*chi_pair))
    return result


def face_area_oscillator_interval(left: torch.Tensor,
                                  right: torch.Tensor,
                                  dt: float) -> torch.Tensor:
    """Matched centered oscillator for each undirected face-area bond.

    The retained rest carrier rotates W=chi wedge Psi by the exact centered
    chi0 clock.  Consequently the standard harmonic interval below has zero
    DEL residual on that orbit.  Its Hamiltonian contribution is positive,
    and its spatial phase flux is carried only by the six incident faces.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    wl = torch.linalg.cross(left_chi, left_psi, dim=-1)
    wr = torch.linalg.cross(right_chi, right_psi, dim=-1)
    result = left.new_zeros((), dtype=torch.float64)
    normalization = native_time.CHI0**2
    for axis in range(3):
        shifts = [0, 0, 0]
        shifts[axis] = -1
        zl = torch.roll(wl, shifts=tuple(shifts), dims=(0, 1, 2))-wl
        zr = torch.roll(wr, shifts=tuple(shifts), dims=(0, 1, 2))-wr
        kinetic = torch.sum(norm2(zr-zl))/(2*dt*normalization)
        potential = (dt*native_time.CHI0**2
                     * torch.sum(norm2(zl)+norm2(zr))
                     /(4*normalization))
        result = result+kinetic-potential
    return result


def face_frame_oscillator_interval(left: torch.Tensor,
                                   right: torch.Tensor,
                                   dt: float,
                                   include_direct_chi: bool = True) -> torch.Tensor:
    """Positive six-face oscillator for the complete native local frame.

    Psi and W=chi wedge Psi carry the retained chi0 rest clock.  Chi is the
    static companion on that orbit, so only its relative face velocity enters.
    This is the smallest reciprocal extension of the area-only candidate that
    communicates all three native prongs without a prepared ray or direction.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    left_w = torch.linalg.cross(left_chi, left_psi, dim=-1)
    right_w = torch.linalg.cross(right_chi, right_psi, dim=-1)
    result = left.new_zeros((), dtype=torch.float64)
    for axis in range(3):
        shifts = [0, 0, 0]
        shifts[axis] = -1

        def bond(value):
            return torch.roll(value, shifts=tuple(shifts),
                              dims=(0, 1, 2))-value

        psi_left, psi_right = bond(left_psi), bond(right_psi)
        chi_left, chi_right = bond(left_chi), bond(right_chi)
        w_left, w_right = bond(left_w), bond(right_w)
        psi_kinetic = torch.sum(norm2(psi_right-psi_left))/(2*dt)
        psi_potential = (dt*native_time.CHI0**2
                         * torch.sum(norm2(psi_left)+norm2(psi_right))/4)
        # The direct chi channel is an ablation.  In the reduced candidate,
        # chi responds reciprocally through W rather than receiving a second,
        # redundant high-inertia broadcast.
        chi_kinetic = (native_time.B*torch.sum(
            norm2(chi_right-chi_left))/(2*dt)
            if include_direct_chi else left.new_zeros((), dtype=torch.float64))
        w_kinetic = (torch.sum(norm2(w_right-w_left))
                     /(2*dt*native_time.CHI0**2))
        w_potential = (dt*torch.sum(
            norm2(w_left)+norm2(w_right))/4)
        result = result+psi_kinetic-psi_potential+chi_kinetic
        result = result+w_kinetic-w_potential
    return result


def parallel_overlap_face_interval(left: torch.Tensor,
                                   right: torch.Tensor,
                                   dt: float) -> torch.Tensor:
    """Six-face oscillator for the parallel part of the native frame.

    The scalar s=chi^dagger Psi/chi0 completes W=chi wedge Psi.  Face
    differences leave the uniform internal carrier untouched.  The matched
    chi0 potential makes the rest-clock sequence an exact DEL solution in
    the activated spatial channel.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)

    def overlap(psi, chi):
        real = torch.sum(
            chi.real*psi.real+chi.imag*psi.imag, dim=-1)
        imag = torch.sum(
            chi.real*psi.imag-chi.imag*psi.real, dim=-1)
        return torch.complex(real, imag)/native_time.CHI0

    def scalar_norm2(value):
        return value.real**2+value.imag**2

    sl, sr = overlap(left_psi, left_chi), overlap(right_psi, right_chi)
    result = left.new_zeros((), dtype=torch.float64)
    for axis in range(3):
        shifts = [0, 0, 0]
        shifts[axis] = -1
        dl = torch.roll(sl, shifts=tuple(shifts), dims=(0, 1, 2))-sl
        dr = torch.roll(sr, shifts=tuple(shifts), dims=(0, 1, 2))-sr
        kinetic = torch.sum(scalar_norm2(dr-dl))/(2*dt)
        potential = (dt*native_time.CHI0**2*torch.sum(
            scalar_norm2(dl)+scalar_norm2(dr))/4)
        result = result+kinetic-potential
    return result


def parallel_overlap_potential(raw: torch.Tensor) -> torch.Tensor:
    side = round((raw.numel()/12)**(1/3))
    psi, chi = local.unpack_tensor(raw, side)
    real = torch.sum(chi.real*psi.real+chi.imag*psi.imag, dim=-1)
    imag = torch.sum(chi.real*psi.imag-chi.imag*psi.real, dim=-1)
    return torch.sum(real**2+imag**2)/2


def parallel_overlap_matched_interval(left: torch.Tensor,
                                      right: torch.Tensor,
                                      dt: float) -> torch.Tensor:
    """Material parallel response completed at the native rest clock."""
    kinetic = (overlap_material.interval(left, right, dt)
               - incumbent.interval(left, right, dt))
    potential = (dt*PARALLEL_MATCHED_POTENTIAL_SCALE
                 *(parallel_overlap_potential(left)
                   + parallel_overlap_potential(right))/2)
    return kinetic-potential


def material_area_matched_interval(left: torch.Tensor,
                                   right: torch.Tensor,
                                   dt: float) -> torch.Tensor:
    """Material completion of the transverse live Psi--chi frame.

    ``chi^dagger Psi / chi0`` and ``chi cross Psi / chi0`` are the two
    coordinate-free pieces of one local three-prong relation.  The parallel
    material candidate transports only the first piece.  This interval
    applies the identical locally inferred velocity to the transverse piece
    and completes its native chi0 rest clock with the matching harmonic
    potential.  No carrier ray, face label, copy, or target velocity occurs.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    velocity = overlap_material.common.best_matching_velocity(
        psi, psi_dot, chi, chi_dot, "both")

    area_left = (torch.linalg.cross(left_chi, left_psi, dim=-1)
                 / native_time.CHI0)
    area_right = (torch.linalg.cross(right_chi, right_psi, dim=-1)
                  / native_time.CHI0)
    area_mid = (area_left+area_right)/2
    area_dot = (area_right-area_left)/dt
    material_area = area_dot + overlap_material.common.advect_same_components(
        area_mid, velocity)
    kinetic = dt*torch.sum(norm2(material_area))/2
    potential = (dt*native_time.CHI0**2
                 * torch.sum(norm2(area_left)+norm2(area_right))/4)
    return kinetic-potential


def factorized_face_potential(raw: torch.Tensor) -> torch.Tensor:
    """Positive cubic face-loop energy derived from the native W field.

    Every elementary square compares its two opposite-corner factorizations.
    The retained carrier makes that defect vanish, so this term has zero
    energy and zero first variation on the rest orbit.  Away from rest its
    Hessian communicates the missing orientation around a complete face.
    """
    side = round((raw.numel()/12)**(1/3))
    psi, chi = local.unpack_tensor(raw, side)
    w = torch.linalg.cross(chi, psi, dim=-1)/native_time.CHI0
    result = raw.new_zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        shift_first = tuple(1 if axis == first else 0 for axis in range(3))
        shift_second = tuple(1 if axis == second else 0 for axis in range(3))
        wa = torch.roll(w, shifts=shift_first, dims=(0, 1, 2))
        wb = torch.roll(w, shifts=shift_second, dims=(0, 1, 2))
        wab = torch.roll(
            w,
            shifts=tuple(shift_first[axis]+shift_second[axis]
                         for axis in range(3)),
            dims=(0, 1, 2))
        defect = (w[..., :, None]*wab[..., None, :]
                  - wa[..., :, None]*wb[..., None, :])
        result = result+torch.sum(
            torch.real(torch.conj(defect)*defect))/2
    return result


def spacetime_factorized_face_interval(left: torch.Tensor,
                                       right: torch.Tensor,
                                       dt: float) -> torch.Tensor:
    """Positive temporal completion of the factorized square-face frame.

    Opposite-corner products are compared across the two committed layers.
    Subtracting the reverse-time product isolates how the factorized face is
    turning, rather than adding another position-only stiffness.  The exact
    carrier makes both cross-layer factorizations vanish independently.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    wl = torch.linalg.cross(left_chi, left_psi, dim=-1)/native_time.CHI0
    wr = torch.linalg.cross(right_chi, right_psi, dim=-1)/native_time.CHI0
    result = left.new_zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        shift_first = tuple(1 if axis == first else 0 for axis in range(3))
        shift_second = tuple(1 if axis == second else 0 for axis in range(3))
        shift_both = tuple(shift_first[axis]+shift_second[axis]
                           for axis in range(3))

        def shifted(value, shift):
            return torch.roll(value, shifts=shift, dims=(0, 1, 2))

        forward = (wl[..., :, None]
                   * shifted(wr, shift_both)[..., None, :]
                   - shifted(wl, shift_first)[..., :, None]
                   * shifted(wr, shift_second)[..., None, :])
        backward = (wr[..., :, None]
                    * shifted(wl, shift_both)[..., None, :]
                    - shifted(wr, shift_first)[..., :, None]
                    * shifted(wl, shift_second)[..., None, :])
        temporal_defect = forward-backward
        result = result+torch.sum(
            torch.real(torch.conj(temporal_defect)*temporal_defect))/(4*dt)
    return result


def symplectic_factorized_face_interval(left: torch.Tensor,
                                        right: torch.Tensor) -> torch.Tensor:
    """Berry one-form of the native square-face factorization defect.

    Im(defect^dagger d defect) is the minimal phase-invariant symplectic
    response of the complete face loop.  It and its first variation vanish
    on the exactly factorized rest carrier.  No prepared direction, copy, or
    target phase occurs in the definition.
    """
    side = round((left.numel()/12)**(1/3))
    left_psi, left_chi = local.unpack_tensor(left, side)
    right_psi, right_chi = local.unpack_tensor(right, side)
    wl = torch.linalg.cross(left_chi, left_psi, dim=-1)/native_time.CHI0
    wr = torch.linalg.cross(right_chi, right_psi, dim=-1)/native_time.CHI0
    result = left.new_zeros((), dtype=torch.float64)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        shift_first = tuple(1 if axis == first else 0 for axis in range(3))
        shift_second = tuple(1 if axis == second else 0 for axis in range(3))
        shift_both = tuple(shift_first[axis]+shift_second[axis]
                           for axis in range(3))

        def defect(value):
            va = torch.roll(value, shifts=shift_first, dims=(0, 1, 2))
            vb = torch.roll(value, shifts=shift_second, dims=(0, 1, 2))
            vab = torch.roll(value, shifts=shift_both, dims=(0, 1, 2))
            return value[..., :, None]*vab[..., None, :] \
                - va[..., :, None]*vb[..., None, :]

        dl, dr = defect(wl), defect(wr)
        midpoint = (dl+dr)/2
        increment = dr-dl
        result = result+torch.sum(torch.imag(torch.conj(midpoint)*increment))
    return result


def interval(left: torch.Tensor, right: torch.Tensor, dt: float) -> torch.Tensor:
    # ``one_form`` is a continuum Lagrangian density (velocity times field),
    # so the variational interval contains dt*one_form.  Omitting dt changes
    # the physical coefficient whenever the integrator is refined.
    result = (incumbent.interval(left, right, dt)
              + dt*SCALE*one_form(left, right, dt))
    if LINK_SOURCE == "joint_face_laplacian_plus_area_oscillator":
        result = result+SCALE*face_area_oscillator_interval(left, right, dt)
    elif LINK_SOURCE == "joint_face_laplacian_plus_frame_oscillator":
        result = result+SCALE*face_frame_oscillator_interval(left, right, dt)
    elif LINK_SOURCE in (
            "joint_face_laplacian_plus_psi_area_oscillator",
            "joint_face_laplacian_plus_psi_area_oscillator_frame_phase",
            "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy"):
        result = result+SCALE*face_frame_oscillator_interval(
            left, right, dt, include_direct_chi=False)
    elif LINK_SOURCE == "joint_face_laplacian_plus_triangle_holonomy":
        # The triangle module is itself one real reciprocal C19 action.  Its
        # incumbent subtraction avoids counting the common base twice; the
        # fixed coefficient is its derived native 1/chi0 normalization.
        result = result+(triangle.interval(left, right, dt)
                         - incumbent.interval(left, right, dt))
    if LINK_SOURCE == (
            "joint_face_laplacian_plus_psi_area_oscillator_triangle_holonomy"):
        result = result+(triangle.interval(left, right, dt)
                         - incumbent.interval(left, right, dt))
    if PLAQUETTE_SCALE != 0:
        result = result-dt*PLAQUETTE_SCALE*(
            factorized_face_potential(left)
            + factorized_face_potential(right))/2
    if SPACETIME_PLAQUETTE_SCALE != 0:
        result = result+SPACETIME_PLAQUETTE_SCALE*(
            spacetime_factorized_face_interval(left, right, dt))
    if SYMPLECTIC_PLAQUETTE_SCALE != 0:
        result = result+SYMPLECTIC_PLAQUETTE_SCALE*(
            symplectic_factorized_face_interval(left, right))
    if PARALLEL_OVERLAP_SCALE != 0:
        # Complete W=chi wedge Psi with the missing parallel scalar
        # s=chi^dagger Psi.  Subtract the shared incumbent so this adds only
        # the positive, rest-null material overlap kinetic energy.
        result = result+PARALLEL_OVERLAP_SCALE*(
            overlap_material.interval(left, right, dt)
            - incumbent.interval(left, right, dt))
    if PARALLEL_FACE_SCALE != 0:
        result = result+PARALLEL_FACE_SCALE*parallel_overlap_face_interval(
            left, right, dt)
    if PARALLEL_MATCHED_SCALE != 0:
        result = result+PARALLEL_MATCHED_SCALE*(
            parallel_overlap_matched_interval(left, right, dt))
    if TRIANGLE_DELTA_SCALE != 0:
        result = result+TRIANGLE_DELTA_SCALE*(
            triangle.interval(left, right, dt)
            - incumbent.interval(left, right, dt))
    if MATERIAL_AREA_SCALE != 0:
        result = result+MATERIAL_AREA_SCALE*material_area_matched_interval(
            left, right, dt)
    return result


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float) -> torch.Tensor:
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
