# Native framework

This document fixes the exact noncanonical substrate candidate used by the
formal chain through `DIRAC-009`. It is a declared starting model, not a claim
that the model is canonical LFM or uniquely selected by deeper physics.

## Fields and constants

At every cubic-lattice site `i` there are exactly two native field families:

\[
\Psi_i\in\mathbb C^3,\qquad \chi_i\in\mathbb C^3.
\]

The fixed constants are

\[
\chi_0=19,\qquad \kappa=\frac1{63},\qquad
B=\frac{\chi_0}{\kappa}=1197.
\]

The C19 shell contains six face neighbors of weight 2 and twelve edge
neighbors of weight 1. There is no independent link, spinor, projector,
matrix, track, or gauge register.

Define

\[
\rho_{\Psi,i}=\Psi_i^\dagger\Psi_i,\qquad
\rho_{\chi,i}=\chi_i^\dagger\chi_i,\qquad
W_i=\chi_i\wedge\Psi_i.
\]

## Action

The Lagrangian is \(L=T-U\), with

\[
T=\frac12\sum_i\lVert\dot\Psi_i\rVert^2+
\frac B2\sum_i\lVert\dot\chi_i\rVert^2.
\]

The local and exterior-area energies are

\[
U_{\rm local}=\sum_i\left[
\frac14(\rho_{\chi,i}-\chi_0^2)^2+
\frac12\rho_{\chi,i}\rho_{\Psi,i}+
\frac14\rho_{\Psi,i}^2\right],
\]

\[
U_{\rm area}=\sum_{i,n\in C19}
\frac{w_n}{2\chi_0^2}\lVert W_{i+n}-W_i\rVert^2.
\]

The native complex-chi covariance is

\[
(C_i\chi_i)_a=\sum_{n\in C19}
\frac{w_n}{8\chi_0^2}(\chi_{i+n})_a
\langle\chi_{i+n},\chi_i\rangle,
\]

and its positive shape energy is

\[
U_{\rm shape}=\frac12\sum_i
\lVert\chi_i\wedge(C_i\chi_i)\rVert^2.
\]

For the six oriented faces let
\(s_{+x,+y,+z}=+1\), \(s_{-x,-y,-z}=-1\), and let all C19 edge signs be zero.
The native incidence residual and its positive energy are

\[
I_i=\sum_{n\in C19}\frac{3s_n}{2\chi_0^2}
\langle\chi_i,\chi_{i+n}\rangle W_{i+n},
\qquad
U_{\rm inc}=\frac12\sum_i\lVert I_i\rVert^2.
\]

The full potential and Hamiltonian are

\[
U=U_{\rm local}+U_{\rm area}+U_{\rm shape}+U_{\rm inc},
\qquad H=T+U.
\]

## Equations and update

Both equations are variations of that one action:

\[
\ddot\Psi_i=-\frac{\delta U}{\delta\Psi_i^\dagger},
\qquad
B\ddot\chi_i=-\frac{\delta U}{\delta\chi_i^\dagger}.
\]

The formal native time map is the synchronous centered recurrence

\[
q_i^{n+1}=2q_i^n-q_i^{n-1}+\Delta t^2 a_i(q^n).
\]

Lean proves that the force families are reciprocal variations of the declared
action and that this update is reversible. The numerical evidence uses the
same action with velocity Verlet to measure finite-step stability; it is a
separate numerical certificate, not part of the Lean theorem.

## Conditional prepared background

The proof assumes an already-running eight-corner complex-\(\Psi\)/complex-\(\chi\)
background. It proves what follows if that state is present; it does not prove
cosmological formation or unique state selection. The precise rays and their
companion map are formalized in `LeanDirac/PreparedBackground.lean`.
