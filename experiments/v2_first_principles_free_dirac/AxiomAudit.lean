module

/-
Compiled-environment audit for the V2 first-principles free-Dirac artifact.

This deliberately runs Lean's environment-level checker after importing the
cumulative theorem.  Unlike a text scan, it cannot be bypassed by comments,
strings, escaped identifiers, declaration modifiers, or imported aliases.
-/

import LeanDirac.V2
import LeanDirac.EnvironmentAudit

#audit_lean_dirac_environment
#print axioms LeanDirac.V2.periodic_full_native_action_generator_intertwines
#print axioms LeanDirac.V2.canonical_spectral_readout_intertwines
#print axioms LeanDirac.V2.canonical_spectral_readout_bijective
#print axioms LeanDirac.V2.canonical_spectral_positive_phase_has_native_density_current
#print axioms LeanDirac.V2.canonical_spectral_centered_update_exact
#print axioms LeanDirac.V2.finite_spectral_native_c19_obeys_position_dirac_with_remainder
#print axioms LeanDirac.V2.finite_c19_current_remainder_tends_to_zero
#print axioms LeanDirac.V2.periodic_integrated_positive_packet_charge_bridge
#print axioms LeanDirac.V2.periodic_integrated_positive_packet_current_bridge
#print axioms LeanDirac.V2.canonical_spectral_native_tangent_is_free_dirac
#print axioms LeanDirac.V2.lfm_canonical_spectral_positive_mode_has_density_current
#print axioms LeanDirac.V2.lfm_canonical_spectral_positive_c19_flux_is_dirac_current
#print axioms LeanDirac.V2.complete_first_principles_free_dirac
#print axioms LeanDirac.V2.cubical_flavor_classification
