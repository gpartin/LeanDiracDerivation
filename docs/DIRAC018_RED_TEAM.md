# DIRAC-018 red-team record

## Question tested

DIRAC-009 introduced a cubical continuum square root. DIRAC-018 asks whether
that root was simply selected because it resembles Dirac, or whether the
already-derived cube geometry forces it. The answer is scoped: it is forced
inside the real, local, first-order, flavor-preserving coefficient class built
from the existing cubical incidence operators and separated mass grading.

## Critical/high findings and repairs

| Severity | Finding | Repair | Disposition |
|---|---|---|---|
| High | The first two-copy definition required each candidate to square on the full eight-corner space, silently tying the copies together. | Replaced it with sector-local square conditions on the exact plus/minus embeddings and proved coefficient uniqueness from an arbitrary nonzero probe in each sector. | Fixed. |
| High | “Unique square root” without a named class would overclaim beyond the theorem. | Put the operator class in the definitions, theorem statement, status, claims, related work, and release guards. Explicitly excluded nonlocal and flavor-mixing roots. | Fixed. |
| High | Cubic covariance and positive orientation might merely select the desired answer unless alternatives were tested. | Proved two controls: the overall negative root still squares and remains covariant but fails orientation; a one-axis reflection still squares but fails cubic covariance. | Fixed. |
| Medium | Two embeddings did not by themselves prove an exact two-copy census. | Proved injectivity, equality of each embedding range with its flavor eigenspace, zero intersection, and decomposition of every eight-corner amplitude. | Fixed. |
| Medium | A continuum classification could be confused with the exact lattice zero structure. | Proved the scalar C19 symbol has only the origin zero in the first Brillouin zone and independently checked periodic grids 2 through 32. Kept exact first-order lattice factorization open. | Fixed. |

No new action term, native field, stencil, runtime projector, Dirac matrix, or
copy-selection rule was introduced.

## Retorts to invalid criticisms

- “This proves every possible lattice Dirac operator is unique.” Retorted:
  the theorem explicitly restricts the candidate class and says nothing about
  arbitrary nonlocal or flavor-mixing roots.
- “The scalar zero census solves fermion doubling.” Retorted: it counts zeros
  of the exact scalar C19 transport symbol; it is not an exact finite-spacing
  first-order Dirac factorization or a universal Nielsen--Ninomiya theorem.
- “Both copies were assumed to use the same coefficients.” Retorted: the final
  two-copy theorem accepts independent candidates and hypotheses on each exact
  embedding, then proves both equal the canonical root.
- “The sign was derived from squaring alone.” Retorted: squaring necessarily
  leaves an overall sign. The sign is selected by the positive mass-frequency
  and current orientation already derived upstream, and the negative control
  makes that dependency explicit.

## Remaining boundary

DIRAC-018 does not prove an exact finite-spacing first-order factorization of
the full C19 symbol, arbitrary-root uniqueness, a universal no-doubling result,
dynamical copy/orientation selection, nonlinear live-chi closure, a localized
stable electron, or electromagnetic coupling. The next mathematical gate is
an exact finite-lattice factorization or a sharply scoped obstruction.
