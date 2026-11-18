import LeanDirac.Dirac022

namespace LeanDirac

/-!
Sole cumulative publication root through DIRAC-022.

The nested certificate starts from the positive two-field C19 action, retains
the exact native second-order and first-order maps, derives the cubical
continuum square root with its finite-lattice remainder, and identifies both
rank-four cubical copies with the conventional Dirac representation, constructs
both frequency graphs, selects the unique nonnegative rest-mass coefficient
from the native running frequency, and proves the prepared-vacuum,
low-momentum continuum evolution by deriving a first-order branch from the
centered recurrence, with its exact native C19 defect and generator limit, and
derives the positive-frequency density/current limit from the same action and
embeddings. The restriction and first variation of the full action are proved
to commute with the retained eight-site embedding. The final certificate states explicitly that the
action-derived Klein--Gordon tangent symbol is the square of the derived
continuum Dirac Hamiltonian.  Nonlinear live-chi closure and a localized
physical electron remain downstream claims.

DIRAC-016 adds a downstream finite-cube theorem: unsigned microscopic
translation exchanges the two retained rank-four copies in one cube
direction and a second translation returns to the original copy.  This does
not change the action or the DIRAC-015 continuum theorem.  It does not assert
that native time evolution performs one exchange per update or that a
localized electron moves by this mechanism.

DIRAC-017 derives the rotation generators from the same signed cube-incidence
operators.  The nonzero rotation algebra fixes their scale uniquely to one
half; the resulting generators are self-adjoint for a positive-definite native
inner product and their Casimir is three quarters.  The same generators are
explicitly intertwined with the conventional four-component Dirac spin
operators on both retained copies, with the two-plus/two-minus component
spectrum exposed.  Their continuous unitary rotations compose, change sign
at `2*pi`, return at `4*pi`, commute with the mass grading, and rotate the full
interpreted mass-plus-transport Hamiltonian covariantly.  The positive native
quarter-turn is uniquely characterized inside the scalar-plus-plane rotor
family by covariance and unitarity.  This is a conditional continuum-sector
spin representation theorem, not a localized-electron or electromagnetic
claim.

DIRAC-018 classifies the continuum square root inside the real, local,
flavor-preserving cubical-incidence coefficient class. Cubic covariance makes
the three transport coefficients a common scalar; exact scalar squaring and
the already selected positive mass/current orientation then force the
canonical coefficients. The proof is sector-local on the two independently
admissible rank-four flavor copies, proves that those copies exhaust the
eight-corner space, and proves that the exact C19 scalar transport symbol has
no additional zero in the first Brillouin zone. Formal negative-orientation
and reflected-axis controls show that the selection and symmetry hypotheses
are necessary. This is not a classification of arbitrary nonlocal or
flavor-mixing roots, nor a theorem about an exact finite-spacing Dirac
operator or a localized electron.

DIRAC-019 constructs a nonzero positive-frequency mode at every momentum and
lifts that branch to arbitrary finite spectral families. Stable evolution on
the recurrence-derived continuum branch preserves every mode's
density for every update count, and the summed four-component current remains
exactly the density-weighted group-velocity moment at every update. Every massive mode is strictly
subluminal, and the finite-family recurrence generator converges to the same
Dirac generator. This is a finite spectral transport theorem, not yet a
real-space localization theorem, finite-time uniform error estimate, or
nonlinear physical-electron transport claim.

DIRAC-020 constructs the missing coherent real-space bridge.  An explicit
normalized Fourier frame is built for every finite abelian periodic lattice,
with orthonormality proved from character orthogonality.  On a periodic cubic
lattice, each character's one-site phase is proved equal to the phase of the
momentum stored in its positive-frequency mode; the spatial and dynamical
mode labels therefore cannot be paired arbitrarily.  The synthesized
four-component field obeys exact Parseval density conservation and its
integrated current is the same density-weighted transport velocity proved in
DIRAC-019.  A nonzero velocity moment therefore gives nonzero spatial current
at every stable update.  The recurrence-versus-continuum field error has an
explicit finite-update pointwise bound.  This is not an exact native
finite-spacing factorization, a uniform fixed-time convergence rate, or a
nonlinear localized-electron transport theorem.

DIRAC-021 follows the rest coefficient already selected from the native action
through the zero-momentum four-component Hamiltonian.  It proves explicit
positive- and negative-energy rest states, their componentwise first-order
time equations, opposite exact phases, and the doubled relative phase of a
coherent branch pair.  In native units the rest angular frequency is `19`, the
relative branch frequency is `38`, the inverse-mass scale is `1/19`, and the
native-coordinate reduced Compton length is `sqrt 12 / 19`.  A pure branch's
global phase is not asserted to be an observable clock; SI mass conversion,
an explicit zitterbewegung observable, electromagnetic coupling, and a
localized nonlinear electron remain downstream.

DIRAC-022 derives the free-continuum zitterbewegung observable without adding
native dynamics.  The velocity is the DIRAC-013 Hamiltonian momentum response,
the mass and rest phases are inherited from DIRAC-011/021, and the normalized
coherent positive/negative rest superposition is proved to have velocity
`c_eff * cos (2 * chi0 * t)`.  Its integrated position amplitude is
`sqrt 12 / 38`, exactly half the native reduced Compton length, while the
peak-to-peak excursion equals that length.  Pure branches, same-energy-sign
superpositions, incoherent mixtures, and a wrong-mass reconstruction are
retained as controls.  This is not a localized native electron trajectory,
an SI prediction, or an electromagnetic result.
-/

structure CumulativeDirac015Certificate : Prop where
  nativeConstants : chi0 = 19 ∧ kappa = 1 / 63 ∧
    chiKineticNormalization = 1197
  nativeC19Weights : ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  freeDiracContinuumLimit : LFMFreeDiracContinuumCertificate

theorem current_highest_gate_is_dirac015 :
    CumulativeDirac015Certificate where
  nativeConstants := ⟨rfl, rfl, chi_kinetic_normalization_eq_1197⟩
  nativeC19Weights := by
    intro neighbor
    fin_cases neighbor <;> norm_num [c19Weight]
  freeDiracContinuumLimit := lfm_free_dirac_continuum_limit

structure CumulativeDirac016Certificate : Prop where
  completedDiracDerivation : CumulativeDirac015Certificate
  copyTranslationStructure : Dirac016CopyTranslationCertificate

theorem current_highest_gate_is_dirac016 :
    CumulativeDirac016Certificate where
  completedDiracDerivation := current_highest_gate_is_dirac015
  copyTranslationStructure := dirac016_exact_copy_translation_structure

structure CumulativeDirac017Certificate : Prop where
  completedDiracDerivationAndCopyStructure : CumulativeDirac016Certificate
  conditionalCubicalSpinHalfRepresentation :
    Dirac017ClosedSpinHalfRepresentationCertificate

theorem current_highest_gate_is_dirac017 :
    CumulativeDirac017Certificate where
  completedDiracDerivationAndCopyStructure :=
    current_highest_gate_is_dirac016
  conditionalCubicalSpinHalfRepresentation :=
    dirac017_closed_spin_half_representation

structure CumulativeDirac018Certificate : Prop where
  completedDiracDerivationCopyAndSpin : CumulativeDirac017Certificate
  nativeCubicalFactorizationClassification :
    Dirac018NativeFactorizationClassificationCertificate

theorem current_highest_gate_is_dirac018 :
    CumulativeDirac018Certificate where
  completedDiracDerivationCopyAndSpin := current_highest_gate_is_dirac017
  nativeCubicalFactorizationClassification :=
    dirac018_native_factorization_classification

structure CumulativeDirac019Certificate : Prop where
  completedChainThroughFactorization : CumulativeDirac018Certificate
  finitePositiveFrequencyPacketTransport :
    Dirac019FinitePacketTransportCertificate

theorem current_highest_gate_is_dirac019 :
    CumulativeDirac019Certificate where
  completedChainThroughFactorization := current_highest_gate_is_dirac018
  finitePositiveFrequencyPacketTransport :=
    dirac019_finite_packet_transport

structure CumulativeDirac020Certificate : Prop where
  completedChainThroughSpectralTransport : CumulativeDirac019Certificate
  coherentSpatialTransport : Dirac020CoherentSpatialTransportCertificate

theorem current_highest_gate_is_dirac020 :
    CumulativeDirac020Certificate where
  completedChainThroughSpectralTransport := current_highest_gate_is_dirac019
  coherentSpatialTransport := dirac020_coherent_spatial_transport

structure CumulativeDirac021Certificate : Prop where
  completedChainThroughCoherentTransport : CumulativeDirac020Certificate
  actionDerivedRestClock : Dirac021RestClockCertificate

theorem current_highest_gate_is_dirac021 :
    CumulativeDirac021Certificate where
  completedChainThroughCoherentTransport := current_highest_gate_is_dirac020
  actionDerivedRestClock := dirac021_action_derived_rest_clock

structure CumulativeDirac022Certificate : Prop where
  completedChainThroughRestClock : CumulativeDirac021Certificate
  actionDerivedZitterbewegung : Dirac022ZitterbewegungCertificate

theorem current_highest_gate_is_dirac022 :
    CumulativeDirac022Certificate where
  completedChainThroughRestClock := current_highest_gate_is_dirac021
  actionDerivedZitterbewegung := dirac022_action_derived_zitterbewegung

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac015
#print axioms LeanDirac.current_highest_gate_is_dirac016
#print axioms LeanDirac.current_highest_gate_is_dirac017
#print axioms LeanDirac.current_highest_gate_is_dirac018
#print axioms LeanDirac.current_highest_gate_is_dirac019
#print axioms LeanDirac.current_highest_gate_is_dirac020
#print axioms LeanDirac.current_highest_gate_is_dirac021
#print axioms LeanDirac.current_highest_gate_is_dirac022
