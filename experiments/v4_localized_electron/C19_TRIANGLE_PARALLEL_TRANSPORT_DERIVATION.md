# C19 triangle parallel-transport derivation

Status: algebraic research candidate only.  No simulation, action admission,
canonical change, commit, push, or promotion follows from this document.

## 1. Native obstruction at a quiet receiver

For a C19 bond `i--j`, the incumbent area energy is

\[
 U_{ij}=\frac{w_{ij}}{2\chi_0^2}\lVert W_j-W_i\rVert^2,
 \qquad W_i=\chi_i\wedge\Psi_i.
\]

At a quiet receiving site, `Psi_i = 0`.  Its two first variations are

\[
 \delta_{\chi_i}W_i=\delta\chi_i\wedge\Psi_i=0,
 \qquad
 \delta_{\Psi_i}W_i=\chi_i\wedge\delta\Psi_i.
\]

Therefore

\[
 \left.\frac{\delta U_{ij}}{\delta\chi_i^\dagger}\right|_{\Psi_i=0}=0,
 \qquad
 \left.\frac{\delta U_{ij}}{\delta\Psi_i^\dagger}\right|_{\Psi_i=0}
 \ne 0 \quad\hbox{generically}.
\]

The incumbent action can launch `Psi` into a quiet neighbor, but it cannot
turn the neighbor's `chi` until local `Psi` has already arrived.  This is the
exact action-level form of the measured late-receiver response.

## 2. What discrete Dirac transport actually requires

Let `Gamma_a` be the three cube-incidence involutions, `beta` cube parity,
and `F` the geometry-derived flavor involution.  The proved cube algebra is

\[
 \Gamma_a^2=\beta^2=F^2=I,\quad
 \{\Gamma_a,\Gamma_b\}=0\ (a\ne b),\quad
 \{\Gamma_a,\beta\}=0,
\]

\[
 [F,\Gamma_a]=[F,\beta]=0.
\]

Consequently the two rank-four projectors

\[
 P_F^\pm=\frac{I\pm F}{2}
\]

are separately invariant under the free Dirac generator

\[
 H_D(k)=m\beta+\sum_{a=1}^3 d_a(k)\Gamma_a.
\]

In particular, `[F,H_D]=0`.  Free Dirac motion does not require the two
four-component copies to exchange.  A compulsory copy-exchange operator
would anticommute with `F` and would be an extra interaction not present in
the free Dirac operator.

Within either copy define

\[
 P_{a,\rightarrow}=\frac{I+\Gamma_a}{2},\qquad
 P_{a,\leftarrow}=\frac{I-\Gamma_a}{2}.
\]

The exact conditional shift along axis `a` is

\[
 (S_a z)_x=P_{a,\rightarrow}z_{x-e_a}
            +P_{a,\leftarrow}z_{x+e_a},
\]

whose Fourier symbol is

\[
 S_a(k)=\exp(-i k_a\Gamma_a).
\]

With the local mass rotation `C(dt)=exp(-i dt m beta/2)`, the symmetric
coin-shift product obeys

\[
 C(dt)\prod_a S_a(dt\,k_a)C(dt)
 =I-i\,dt\left(m\beta+\sum_a k_a\Gamma_a\right)+O(dt^2).
\]

Thus the required native behavior is not an instantaneous reconstruction of
all eight corners.  It is component-selective transport plus internal
rotation, separately in each flavor copy.

## 3. A triangle quantity that can act before local Psi arrives

Let `T19` contain triples `(i,j,k)` for which all three edges are C19 bonds.
For each oriented triangle define the native Bargmann loop

\[
 B_{i;jk}
  =(\chi_i^\dagger\Psi_j)
   (\Psi_j^\dagger\Psi_k)
   (\Psi_k^\dagger\chi_i).
\]

This scalar has the required properties:

1. it is made only from native `Psi`, native `chi`, and a C19 triangle;
2. it is invariant under constant phase changes of either field family;
3. it contains the relative phase and orientation of the two sender sites;
4. it contains no corner label, prepared ray, spinor, Dirac matrix, target
   direction, or receiving-site `Psi`;
5. it is conjugated when the triangle orientation is reversed.

Most importantly, even when `Psi_i=0`,

\[
 \delta_{\chi_i}B_{i;jk}
  =(\delta\chi_i^\dagger\Psi_j)(\Psi_j^\dagger\Psi_k)
    (\Psi_k^\dagger\chi_i)
   +(\chi_i^\dagger\Psi_j)(\Psi_j^\dagger\Psi_k)
    (\Psi_k^\dagger\delta\chi_i),
\]

which is generically nonzero.  A triangle loop can therefore orient the
receiving `chi` before local `Psi` has appreciable amplitude.  The incumbent
two-site area energy cannot do this.

## 4. Smallest phase-lock and gyroscopic completion

Two real triangle invariants follow directly from `B`:

\[
 U_{\rm phase}
  =\frac{\lambda}{\chi_0^8}
    \sum_{(i,j,k)\in T_{19}}[\operatorname{Im}B_{i;jk}]^2,
\]

\[
 L_{\rm hol}
  =\frac{\mu}{\chi_0^8}
    \sum_{(i,j,k)\in T_{19}}\epsilon_{ijk}
      \operatorname{Im}(\overline{B}_{i;jk}\,\dot B_{i;jk}).
\]

Here `epsilon_ijk` is the orientation of the spatial C19 triangle.  Under a
spatial reflection both `epsilon` and the loop orientation reverse, so the
full term is a cubic scalar.  Summing every cubic image removes any preferred
axis.

`U_phase` is nonnegative and locks the loop phase to either `0` or `pi`.
It does not delete either chirality.  The two choices are compatible with the
two preserved flavor copies.

The one-form in `L_hol` is not a total derivative.  Writing `B=x+i y`,

\[
 \operatorname{Im}(\overline B\,dB)=x\,dy-y\,dx,
 \qquad
 d[\operatorname{Im}(\overline B\,dB)]=2\,dx\wedge dy\ne0.
\]

It therefore supplies genuine symplectic curvature and can change timing,
rather than merely adding another static stiffness.  Because `dot B`
contains `dot chi_i`, `dot Psi_j`, and `dot Psi_k`, variation produces the
reciprocal force in both field equations.

The research action class is

\[
 L_{\rm research}=L_{\rm V40}+L_{\rm hol}-U_{\rm phase}.
\]

The gyroscopic one-form changes canonical momenta but contributes no separate
potential energy.  The Hamiltonian remains

\[
 H_{\rm research}=H_{\rm V40}+U_{\rm phase}
\]

after the corresponding momentum shift, so it is bounded below whenever the
V40 Hamiltonian is bounded below and `lambda >= 0`.

## 5. Reciprocal Euler--Lagrange structure

For all real native coordinates collected into `q`, write

\[
 L=\tfrac12\dot q^T M\dot q+A(q)^T\dot q-U(q).
\]

The triangle one-form supplies `A`.  The equations are

\[
 M\ddot q+\bigl(\partial A-(\partial A)^T\bigr)\dot q+\nabla U=0.
\]

The middle matrix is antisymmetric.  It redirects motion without doing work:

\[
 \dot q^T\bigl(\partial A-(\partial A)^T\bigr)\dot q=0.
\]

This is the precise action mechanism by which incoming phase flow can turn
the receiving `chi` without an opposing holding force or energy injection.

## 6. Compression requirements

Let `E_+` and `E_-` be the two proved four-component cube embeddings.  An
admissible triangle connection must satisfy, at the localized rest orbit,

\[
 E_\pm^\dagger K_{\rm hol}(k)E_\mp=0,
\]

so that it does not bolt on flavor conversion, and

\[
 E_\pm^\dagger K_{\rm hol}(k)E_\pm
   =\pm\sum_a v_a k_a\Gamma_a+O(|k|^2).
\]

The opposite signs are a possible native role for the two copies: opposite
triangle holonomy while each remains a complete Dirac carrier.  The speeds
`v_a` must be derived from the native Hessian/symplectic form, not fitted to
Dirac.

The quiet-receiver test is already passed algebraically because
`delta_chi B != 0` does not require `Psi_i`.  The copy-preserving transport
condition, exact rest orbit, coefficient provenance, and response rank are
open.  Those are algebraic gates to close before any simulation.

## 7. Verdict

The derivation rejects the naive claim that the two copies must alternate as
leader and follower.  It identifies a smaller native mechanism:

* C19 triangles carry a phase-sensitive holonomy;
* that holonomy can torque quiet receiving `chi` before local `Psi` arrives;
* an antisymmetric triangle one-form can change timing without adding or
  removing energy;
* the two flavor copies may carry opposite holonomy while remaining
  dynamically separate.

This is a derived action class, not a passing action.  `lambda` and `mu` are
not yet selected, and no numerical evolution has been run.
