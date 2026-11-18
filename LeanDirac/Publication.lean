import LeanDirac.Dirac019

namespace LeanDirac

/-!
Sole cumulative publication root through DIRAC-019.

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

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac015
#print axioms LeanDirac.current_highest_gate_is_dirac016
#print axioms LeanDirac.current_highest_gate_is_dirac017
#print axioms LeanDirac.current_highest_gate_is_dirac018
#print axioms LeanDirac.current_highest_gate_is_dirac019
