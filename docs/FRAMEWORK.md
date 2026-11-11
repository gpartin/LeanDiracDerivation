# Native framework

This document fixes the noncanonical substrate candidate used by the formal
chain through DIRAC-010. It is a declared starting model, not a claim that the
model is canonical LFM or uniquely selected by deeper physics.

## Fields and constants

At every cubic-lattice site i there are exactly two native field families:

[
\Psi_i\in\mathbb C^3,\qquad \chi_i\in\mathbb C^3.
]

The fixed constants are

[
\chi_0=19,\qquad \kappa=\frac1{63},\qquad
B=\frac{\chi_0}{\kappa}=1197.
]

C19 contains six face neighbors of weight 2 and twelve edge neighbors of
weight 1. There is no independent link, spinor, projector, matrix, track, or
gauge register.

Define

[
\rho_{\Psi,i}=\Psi_i^\dagger\Psi_i,\qquad
\rho_{\chi,i}=\chi_i^\dagger\chi_i,\qquad
W_i=\chi_i\wedge\Psi_i.
]

## Action

The Lagrangian is L=T-U, with

[
T=\frac12\sum_i\lVert\dot\Psi_i\rVert^2+
\frac B2\sum_i\lVert\dot\chi_i\rVert^2,
]

[
U_{\rm local}=\sum_i\left[
\frac14(\rho_{\chi,i}-\chi_0^2)^2+
\frac12\rho_{\chi,i}\rho_{\Psi,i}+
\frac14\rho_{\Psi,i}^2\right],
]

[
U_{\rm area}=\sum_{i,n\in C19}
\frac{w_n}{2\chi_0^2}\lVert W_{i+n}-W_i\rVert^2,
\qquad U=U_{\rm local}+U_{\rm area}.
]

No shape-holding, signed-incidence, gamma, beta, or Dirac term appears in the
native action.

## Equations and update

Both equations are variations of the single action:

[
\ddot\Psi_i=-\frac{\delta U}{\delta\Psi_i^\dagger},\qquad
B\ddot\chi_i=-\frac{\delta U}{\delta\chi_i^\dagger}.
]

The formal native time map is the synchronous centered recurrence

[
q_i^{n+1}=2q_i^n-q_i^{n-1}+\Delta t^2 a_i(q^n).
]

Lean proves the reciprocal variations and the discrete-action update.

## Conditional prepared background

The chain assumes an already-running eight-corner complex-Psi/complex-chi
background. It proves what can run if that state is present; it does not claim
cosmological formation or unique state selection. The rays and companion map
are formalized in `LeanDirac/PreparedBackground.lean`.

## Downstream Dirac readout

The cubical Clifford operators first appear after the native C19 transport
symbol has been derived. They are a low-momentum square-root readout of its
isotropic quadratic term, not native action terms. Lean retains the exact
finite-lattice remainder and proves the standard four-component change of
basis for both native copies.
