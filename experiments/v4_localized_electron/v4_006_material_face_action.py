"""Research replacement: material derivative of the existing face oscillator.

No new interaction or parameter is appended. The existing direct-prong
action already derives one common native velocity v. Its material rate
D_t F = d_t F + A_v F is used inside the existing positive six-face
oscillator kinetic square, instead of the laboratory rate d_t F. The
matched native clock potential, C19 weights, B and other terms stay intact.
Three variants diagnose whether the Psi, W or complete frame is overcounting
translation as internal oscillator velocity. All are unadmitted.
"""
from __future__ import annotations

import numpy as np
import torch

import density_mold_connection_probe as local
import exact_translation_multichannel_prong_gate as common
import live_clifford_bond_action as incumbent


def delta_interval(left,right,dt,mode):
    horizontal=mode.startswith('covariant_')
    if horizontal:mode=mode.removeprefix('covariant_')
    side=round((left.numel()/12)**(1/3));lp,lc=local.unpack_tensor(left,side);rp,rc=local.unpack_tensor(right,side)
    p=(lp+rp)/2;c=(lc+rc)/2;pd=(rp-lp)/dt;cd=(rc-lc)/dt
    velocity=common.best_matching_velocity(p,pd,c,cd,'both')
    clock=incumbent.native_time.CHI0
    lw=torch.linalg.cross(lc,lp,dim=-1)/clock;rw=torch.linalg.cross(rc,rp,dim=-1)/clock
    w=(lw+rw)/2;wd=(rw-lw)/dt
    fields=[]
    if mode in ('Psi','Psi_W','all'):fields.append((p,pd,1.0,True))
    if mode in ('W','Psi_W','all'):fields.append((w,wd,1.0,True))
    if mode in ('chi','all'):fields.append((c,cd,incumbent.native_time.B,False))
    result=left.new_zeros(())
    for q,d,weight,rotates in fields:
        adv=common.advect_same_components(q,velocity)
        if horizontal and rotates:
            # Matched endpoint potential yields the exact midpoint kinetic
            # scale cos^2(theta/2). Retain the incumbent gyro/clock pieces,
            # changing only the positive square of the shape rate.
            theta=float(np.arccos(1-dt**2*clock**2/2))
            weight*=np.cos(theta/2)**2
            d=d-1j*(2*np.tan(theta/2)/dt)*q
        for axis in range(3):
            u=torch.roll(d,-1,axis)-d;v=torch.roll(adv,-1,axis)-adv
            # Exact difference of the material and ordinary kinetic squares.
            result=result+weight*torch.sum(torch.real(u.conj()*v)+torch.real(v.conj()*v)/2)*dt
    return incumbent.SCALE*result


def interval(left,right,dt,mode):return incumbent.interval(left,right,dt)+delta_interval(left,right,dt,mode)


def delta_del(previous,current,following,dt,mode):
    def summed(p,q,r):return delta_interval(p,q,dt,mode)+delta_interval(q,r,dt,mode)
    return torch.func.grad(summed,argnums=1)(previous,current,following)


def del_residual(previous,current,following,dt,mode):
    return incumbent.del_residual(previous,current,following,dt)+delta_del(previous,current,following,dt,mode)
