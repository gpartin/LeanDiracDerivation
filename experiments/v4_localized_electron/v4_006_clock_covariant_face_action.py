"""Research alternative: make the existing face oscillator a shape metric.

The existing matched face oscillator stores essentially all native phase
momentum in its spatial bond norms. Replace (not add to) a selected matched
oscillator by the same positive squared clock-covariant bond increment
already used in the direct-prong communication. Both old and new terms
annihilate the exact rest orbit; the replacement removes the spurious
rest-phase charge of that oscillator. Coefficient B and C19 are unchanged.
This is an unadmitted action comparison, not a new electron-specific force.
"""
from __future__ import annotations

import numpy as np
import torch

import density_mold_connection_probe as local
import live_clifford_bond_action as incumbent


def delta_interval(left: torch.Tensor, right: torch.Tensor, dt: float,
                   mode: str) -> torch.Tensor:
    side=round((left.numel()/12)**(1/3));lp,lc=local.unpack_tensor(left,side);rp,rc=local.unpack_tensor(right,side)
    clock=incumbent.native_time.CHI0;theta=float(np.arccos(1-dt**2*clock**2/2));e=np.exp(1j*theta)
    quantities=[]
    if mode in ('Psi','both'):quantities.append((lp,rp))
    if mode in ('W','both'):quantities.append((torch.linalg.cross(lc,lp,dim=-1)/clock,torch.linalg.cross(rc,rp,dim=-1)/clock))
    total=left.new_zeros(())
    for l,r in quantities:
        for axis in range(3):
            x=torch.roll(l,-1,axis)-l;y=torch.roll(r,-1,axis)-r
            # Existing matched oscillator, factored at its own centered rest
            # phase. Difference cancels it and retains the same kinetic scale.
            def sq(z):return torch.sum(torch.real(z.conj()*z))
            old=(sq(y-x)/(2*dt)-dt*clock**2*(sq(x)+sq(y))/4)
            new=sq(y-e*x)/(2*dt)
            total=total+new-old
    return incumbent.SCALE*total


def interval(left,right,dt,mode):
    return incumbent.interval(left,right,dt)+delta_interval(left,right,dt,mode)


def del_residual(previous,current,following,dt,mode):
    def summed(p,q,r):return interval(p,q,dt,mode)+interval(q,r,dt,mode)
    return torch.func.grad(summed,argnums=1)(previous,current,following)
