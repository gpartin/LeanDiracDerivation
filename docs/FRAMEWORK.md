# Declared native framework

The derivation starts from this discrete model. It does not assume a Dirac
field, Dirac matrix, spinor rule, selected cube ray, projector, or maintained
background.

## Fields and geometry

At every periodic cubic-lattice site `i`:

\[
\Psi_i\in\mathbb C^3,\qquad \chi_i\in\mathbb C^3.
\]

Let

\[
\rho_{\Psi,i}=\Psi_i^\dagger\Psi_i,\qquad
\rho_{\chi,i}=\chi_i^\dagger\chi_i,\qquad
W_i=\chi_i\wedge\Psi_i.
\]

C19 contains six face neighbors with relative weight 2 and twelve edge
neighbors with relative weight 1. The constants are

\[
\chi_0=19,\qquad \kappa=\frac1{63},\qquad B=\frac{\chi_0}{\kappa}=1197.
\]

## Action

\[
L=T-U,
\]

\[
T=\frac12\sum_i\|\dot\Psi_i\|^2+
\frac B2\sum_i\|\dot\chi_i\|^2,
\]

\[
U_{\rm local}=\sum_i\left[
\frac14(\rho_{\chi,i}-\chi_0^2)^2+
\frac12\rho_{\chi,i}\rho_{\Psi,i}+
\frac14\rho_{\Psi,i}^2\right],
\]

\[
U_{\rm area}=\sum_{(i,n)\in C19}
\frac{w_n}{2\chi_0^2}\|W_{i+n}-W_i\|^2,
\qquad U=U_{\rm local}+U_{\rm area}.
\]

The native equations are derived by variation:

\[
\ddot\Psi_i=-\frac{\delta U}{\delta\Psi_i^\dagger},\qquad
B\ddot\chi_i=-\frac{\delta U}{\delta\chi_i^\dagger}.
\]

The native synchronous centered update is

\[
q_i^{n+1}=2q_i^n-q_i^{n-1}+\Delta t^2 a_i(q^n).
\]

The Dirac operator appears only after the action variation, vacuum proof,
finite-lattice Hessian, phase-space construction, and cubical incidence root.
It is not a third native equation.
