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
\chi_0=19,\qquad B=1,\qquad \kappa=\frac{\chi_0}{B}=19.
\]

Here `chi0 = 19` is the C19 mode count: one center, six faces, and twelve
edges. V3.6 selects one common Euclidean kinetic metric for the two cubical
grade representations and computes the *relative* coefficient from their
unit-blade norm ratio. In plane order
`(01,02,12)`, the Euclidean Hodge map sends vector coordinates `(x,y,z)` to
bivector coordinates `(z,-y,x)`. Lean proves this map is invertible and
norm-preserving. Constructing both kinetic terms from this one standard
Euclidean cubical Clifford metric assigns corresponding unit vector and unit
bivector blades equal weight, fixing `B = 1` and therefore `kappa = 19`.

This does not identify or combine the two physical fields. Both remain native
`C^3` arrays. `Psi` is represented in the vector grade, while `chi`'s three
coordinates are mapped to the bivector grade for the metric comparison. It
selects one shared kinetic metric for those representations. The historical pair
`(kappa,B)=(1/63,1197)` and the C19-weight ablation `(1,19)` are retained as
controls, not as live action values. No measured constant or mobility result
is used to choose `B = 1`.

The free-periodic transverse theorem is proved for arbitrary `B`, and at the
selected exact vacuum its linear `Psi` sector has a vanishing `Psi`--`chi`
cross Hessian. Consequently the complete free-Dirac chain survives the V3.6
action change. This does not make `B` irrelevant: it changes live-`chi`
inertia, nonlinear response, localization, and future moving-electron tests.

## Action

\[
L=T-U,
\]

\[
T=\frac12\sum_i\|\dot\Psi_i\|^2+
\frac12\sum_i\|\dot\chi_i\|^2,
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
 \ddot\chi_i=-\frac{\delta U}{\delta\chi_i^\dagger}.
\]

The native synchronous centered update is

\[
q_i^{n+1}=2q_i^n-q_i^{n-1}+\Delta t^2 a_i(q^n).
\]

The Dirac operator appears only after the action variation, vacuum proof,
finite-lattice Hessian, phase-space construction, and cubical incidence root.
It is not a third native equation.
