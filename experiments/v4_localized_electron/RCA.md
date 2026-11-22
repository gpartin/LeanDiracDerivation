# V4-002 root-cause analysis

**Issue:** The exact localized V40 rest orbit is linearly unstable under the
complete native two-layer tangent evolution. The best bounded research action
tested so far greatly reduces but does not remove the growth.

**Why it matters:** V4 requires the same localized carrier to survive before
its four-component readout, spin, translation, density, and current can be
identified with one physical electron. An exact orbit is not sufficient when
arbitrarily small perturbations grow at a finite physical rate.

**Symptoms:** On the unchanged action (`B = 1197`), the largest Floquet
multiplier gives growth rates `1.041396`, `1.038175`, and `1.037373` at
`dt = 1/128`, `1/256`, and `1/512`. The result is identical on side-three and
side-four periodic boxes. The leading mode is completely supported on the
eight carrier sites and is an independent corner amplitude/phase shear. It is
not a translation mode, exterior radiation, boundary reflection, or timestep
artifact.

**Expected symptoms:** A passing rest orbit has no finite positive growth rate
as the timestep is refined, preserves reciprocity and energy structure, and
remains stable when the box is enlarged.

**Root cause:** The canonical local density response and the single quadratic
bond-area response do not provide enough independent curvature to stabilize
all internal corner directions. A bounded generic bond polynomial can suppress
the original amplitude/phase mode, but the dominant residual then becomes a
transverse third-component `chi` mode. The local density, density-memory, and
parallel-overlap invariants have zero or adverse response in that direction.
A bounded onsite polynomial in `|chi wedge Psi|^2` controls that transverse
direction but exposes a larger in-plane corner mode instead; it does not close
the full spectrum. This is an action-Hessian rank deficiency for this
localized orbit, not a bad launch or an error in the exact orbit theorem.

**Corrective action:** Derive the next action basis from the unresolved
transverse `chi` Hessian channel, subject to locality, C19/cubic symmetry,
reciprocity, bounded energy, and unchanged low-amplitude C19 propagation.
Calculate its response rank before another coefficient sweep. Do not promote
the current diagnostic polynomial or its provisional `B = 229`: their
coefficients lack geometric provenance and their converged growth remains
positive.
