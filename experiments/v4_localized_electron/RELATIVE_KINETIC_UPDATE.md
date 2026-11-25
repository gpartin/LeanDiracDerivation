# Relative-kinetic research action and derived update

This note derives the update used to test the noncanonical research term. It
does not modify or promote the canonical V40 action.

Let

```text
A_i = chi_i x PsiDot_i - chiDot_i x Psi_i

T_eta = 1/2 sum_i |PsiDot_i|^2
      + B/2 sum_i |chiDot_i|^2
      + eta/(2 chi0^2) sum_i |A_i|^2.
```

The canonical momenta are

```text
p_Psi = PsiDot + eta/chi0^2 (A x chi*)
p_chi = B chiDot + eta/chi0^2 (A x Psi*).
```

The position derivative of the kinetic energy is

```text
T_Psi = eta/chi0^2 (chiDot* x A)
T_chi = eta/chi0^2 (PsiDot* x A).
```

For adjacent layers `q_a,q_b`, define

```text
q_bar(a,b) = (q_a + q_b)/2
v(a,b)     = (q_b - q_a)/dt.
```

The self-adjoint discrete Lagrangian is

```text
Ld(q_a,q_b) = dt T_eta(q_bar(a,b),v(a,b))
            - dt/2 [U(q_a)+U(q_b)].
```

Varying the sum of two adjacent discrete actions gives the exact implicit
update equation

```text
0 = p(q_{n-1/2},v_{n-1/2}) - p(q_{n+1/2},v_{n+1/2})
  + dt/2 [T_q(q_{n-1/2},v_{n-1/2})
         +T_q(q_{n+1/2},v_{n+1/2})]
  + dt F_U(q_n),
```

where `F_U=-delta U/delta q`. The equation is solved for `q_(n+1)`.
Swapping the two endpoints and reversing velocity gives the same equation, so
the update is time reversible. At `eta=0`, the momenta reduce to the diagonal
native momenta, `T_q=0`, and the equation becomes centered Verlet exactly.

For `eta>=0` the added kinetic square is nonnegative. A negative coefficient
is not globally admissible because sufficiently large field amplitudes can
make the kinetic metric indefinite.

## Numerical verdict

The `eta=0` implementation reproduces the prior analytic Verlet growth rate to
`9.43e-8`. At `eta=0.01` it admits an exact re-solved rest orbit with RMS
Euler--Lagrange residual `3.95e-12`, but the growth rate rises from `0.07556`
to `0.30791`. Negative diagnostic values also worsen growth and are not
globally positive. The symmetric relative-inertia candidate is rejected.
