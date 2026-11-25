"""Stable incumbent plus the native C19 face--edge triangle movement delta.

This is the side-independent form of the reduced triangle action. Its body
endpoint is evaluated as the average of the three C19 face+edge paths. The
positive path energy and common/relative oriented-area Berry forms have the
native rest-torque-cancelling ratio

  a = 2 dt / sin(theta),       body/local = -1/4,

where ``theta`` is the exact centered rest-clock step. No Dirac operator,
carrier label, copy selector, or preferred axis enters the action.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import torch

os.environ.setdefault("LFM_RELATIVE_PRONG_WEIGHT", "0")

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import wake_stabilized_completed_action as incumbent


SIDE = 5
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))
TRIANGLE_SCALE = float(os.environ.get(
    "LFM_TRIANGLE_MOVEMENT_SCALE", str(1/rank.R)))


def fields(raw: torch.Tensor):
    return local.unpack_tensor(raw, SIDE)


def area(psi: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(chi, psi, dim=-1)/rank.R


def midpoint_kinematics(left: torch.Tensor, right: torch.Tensor, dt: float):
    left_psi, left_chi = fields(left)
    right_psi, right_chi = fields(right)
    psi = (left_psi+right_psi)/2
    chi = (left_chi+right_chi)/2
    psi_dot = (right_psi-left_psi)/dt
    chi_dot = (right_chi-left_chi)/dt
    w = area(psi, chi)
    common = (torch.linalg.cross(chi, psi_dot, dim=-1)
              + torch.linalg.cross(chi_dot, psi, dim=-1))/rank.R
    relative = (torch.linalg.cross(chi, psi_dot, dim=-1)
                - torch.linalg.cross(chi_dot, psi, dim=-1))/rank.R
    return w, common, relative


def local_pair(w: torch.Tensor, velocity: torch.Tensor, dt: float):
    return dt*torch.sum(torch.imag(torch.sum(
        torch.conj(w)*velocity, dim=-1)))


def decompositions(body):
    for axis in range(3):
        yield tuple(body[index] if index == axis else 0
                    for index in range(3))


def triangle_pair(w: torch.Tensor, velocity: torch.Tensor, dt: float):
    result = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        endpoint_w_direct = torch.roll(w, shifts=body, dims=(0, 1, 2))
        endpoint_v_direct = torch.roll(
            velocity, shifts=body, dims=(0, 1, 2))
        for face in decompositions(body):
            middle_w = torch.roll(w, shifts=face, dims=(0, 1, 2))
            middle_v = torch.roll(velocity, shifts=face, dims=(0, 1, 2))
            endpoint_w = middle_w+(endpoint_w_direct-middle_w)
            endpoint_v = middle_v+(endpoint_v_direct-middle_v)
            result = result+torch.sum(torch.imag(
                torch.sum(torch.conj(w)*endpoint_v, dim=-1)
                + torch.sum(torch.conj(endpoint_w)*velocity, dim=-1)))
    return dt*result/12


def triangle_energy(raw: torch.Tensor):
    psi, chi = fields(raw)
    w = area(psi, chi)
    result = torch.zeros((), dtype=torch.float64)
    for body in BODY_OFFSETS:
        endpoint = torch.roll(w, shifts=body, dims=(0, 1, 2))
        for face in decompositions(body):
            middle = torch.roll(w, shifts=face, dims=(0, 1, 2))
            difference = (middle-w)+(endpoint-middle)
            result = result+torch.sum(
                difference.real**2+difference.imag**2)
    return result/12


def interval(left: torch.Tensor, right: torch.Tensor, dt: float):
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    coefficient = 2*dt/np.sin(theta)
    w, common, relative = midpoint_kinematics(left, right, dt)
    gyro = coefficient*(
        local_pair(w, common, dt)+local_pair(w, relative, dt)
        - (triangle_pair(w, common, dt)
           + triangle_pair(w, relative, dt))/4)
    potential = dt*(triangle_energy(left)+triangle_energy(right))/2
    delta = -potential+gyro
    return incumbent.interval(left, right, dt)+TRIANGLE_SCALE*delta


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor, dt: float):
    def summed(left, middle, right):
        return interval(left, middle, dt)+interval(middle, right, dt)
    return torch.func.grad(summed, argnums=1)(previous, current, following)
