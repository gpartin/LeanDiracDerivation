"""Conservative transport of the existing face-frame oscillator (research).

The previous material derivative acted on the site fields and differentiated
afterwards. Instead transport the face frame itself with an antisymmetric
neighbor operator. Across each adjacent pair the coefficient is derived
from the existing common native velocity. The smooth conjugate phase link
makes opposite directions reciprocal and kills the rest-clock cross term
locally: Re[(i*F_i)^dagger (A_v F)_i]=0. No extra field, direction selector,
Dirac generator, fitted coefficient or imposed translated state is used.
This research replacement must be rejected if rest/current/transport fail.
"""
from __future__ import annotations

import numpy as np
import torch

import density_mold_connection_probe as local
import exact_translation_multichannel_prong_gate as common
import live_clifford_bond_action as incumbent


def skew_face_advection(frame,velocity,aligned=True):
    result=torch.zeros_like(frame)
    for axis in range(3):
        neighbor=torch.roll(frame,-1,axis);v=.5*(velocity[...,axis]+torch.roll(velocity[...,axis],-1,axis))
        if aligned:
            link=torch.sum(frame.conj()*neighbor,dim=-1)
            # Same already-declared local kappa area scale, not a new fit.
            floor=incumbent.KAPPA*incumbent.native_time.CHI0**2
            u=link.conj()/torch.sqrt(torch.real(link.conj()*link)+floor**2)
        else:u=torch.ones_like(v)
        right=(v*u)[...,None]*neighbor
        reverse=(v*u.conj())[...,None]*frame
        result=result+(right-torch.roll(reverse,1,axis))/2
    return result


def delta_interval(left,right,dt,mode='aligned'):
    side=round((left.numel()/12)**(1/3));lp,lc=local.unpack_tensor(left,side);rp,rc=local.unpack_tensor(right,side)
    p=(lp+rp)/2;c=(lc+rc)/2;pd=(rp-lp)/dt;cd=(rc-lc)/dt
    velocity=common.best_matching_velocity(p,pd,c,cd,'both')
    clock=incumbent.native_time.CHI0
    lw=torch.linalg.cross(lc,lp,dim=-1)/clock;rw=torch.linalg.cross(rc,rp,dim=-1)/clock
    # Exact midpoint decomposition of the existing matched oscillator.
    # Its coefficient on |D_t F|² is cos²(theta/2), the clock potential is
    # unchanged. The direct chi kinetic is not replaced in this probe.
    a=1-dt**2*clock**2/4;total=left.new_zeros(())
    for low,high in ((lp,rp),(lw,rw)):
        for face_axis in range(3):
            fl=torch.roll(low,-1,face_axis)-low;fr=torch.roll(high,-1,face_axis)-high
            mid=(fl+fr)/2;dot=(fr-fl)/dt
            face_velocity=(velocity+torch.roll(velocity,-1,face_axis))/2
            adv=skew_face_advection(mid,face_velocity,mode=='aligned')
            total=total+torch.sum(torch.real(dot.conj()*adv)+torch.real(adv.conj()*adv)/2)
    return incumbent.SCALE*dt*a*total


def interval(left,right,dt,mode='aligned'):return incumbent.interval(left,right,dt)+delta_interval(left,right,dt,mode)
