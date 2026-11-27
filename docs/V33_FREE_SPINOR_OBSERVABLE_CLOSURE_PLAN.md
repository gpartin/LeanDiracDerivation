# V3.3 Free-Spinor Observable Closure Plan

## Requested finish line

V3.3 must make the continuum free-spinor content of the cumulative LFM Dirac
chain directly usable: a complete component basis, positive/negative-energy
states, their resolution of an arbitrary spinor, derived bilinear observables,
helicity information, a narrowly scoped Gordon relation, and the free
momentum-space inverse away from its poles.

V3.3 changes no LFM action, governing equation, stencil, field, coefficient,
vacuum, or readout. It is downstream mathematics of the V3.2 certificate.

## Design red-team preflight

### Required pass gates

1. The exact V3.2 cumulative theorem remains an explicit dependency.
2. Every new definition is built only from already-derived V3.2 objects.
3. Positive and negative branches are eigenstates only under the existing
   on-shell and nonzero-energy hypotheses.
4. Branch completeness is stated for arbitrary four-component states.
5. Observable claims use the existing positive inner product, current, and
   spin operators.
6. The Gordon result is labeled forward/on-shell and is not described as an
   electromagnetic vertex or a proof of the magnetic moment.
7. The propagator is an algebraic free resolvent away from its pole.
8. The cumulative theorem has no `sorry`, added axiom, or hidden physical
   calibration.

### Disallowed live mechanisms

- new action terms or fitted coefficients;
- an external gauge field, electromagnetic interaction, or magnetic moment;
- CAR/Fock-space or occupation axioms;
- SI units, MeV calibration, or the measured electron mass as input;
- packet transport or claims assigned to the separate V4 program;
- a claim that the formal free excitation is already a physical electron.

### Planned live mechanisms

- the V3.2 continuum Hamiltonian and its square;
- V3.2 positive/negative spectral projectors;
- the cube-derived spin generator and positive inner product;
- the normalized positive-frequency current bridge;
- elementary finite-dimensional complex algebra.

### Claim boundary

Passing V3.3 proves a more complete free linearized spin-one-half observable
package derived from the same LFM chain. It does not prove localization,
self-binding, charge quantization, electromagnetic response, transport, or an
absolute physical mass scale.

## Ten meaningful handoffs

1. `V33FrameworkLock`: freeze the cumulative input and no-change boundary.
2. `V33SpinorBasis`: prove the canonical four-component basis resolves every
   spinor and is orthonormal.
3. `V33ProjectedSpinors`: construct both energy branches and prove their
   eigenvalue and completeness properties.
4. `V33SpinSums`: package branch resolution as the exact free spectral sum.
5. `V33BilinearObservables`: expose density/current statements for normalized
   positive-frequency states.
6. `V33Helicity`: prove the cube-derived spin spectrum and an exact
   axis-aligned helicity statement.
7. `V33GordonIdentity`: prove the scoped forward on-shell current identity.
8. `V33FreePropagator`: prove the free resolvent identity off the mass shell.
9. `V33ScopeAudit`: machine-check the inherited successes and explicit
   exclusions needed to prevent claim inflation.
10. `V33CompleteFreeSpinor`: assemble the cumulative V3.3 certificate.

## Verification and review

After all ten modules compile, `scripts/verify_v33_research.py` must build the
cumulative facade, reject unfinished proofs and unapproved axioms, and check
the documented scope. Two hostile red-team reviews follow. Critical and high
findings block completion; straightforward medium and low findings are fixed.
