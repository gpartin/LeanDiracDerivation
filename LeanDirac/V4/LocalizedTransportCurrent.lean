/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.V4.LocalizedSpin
public import LeanDirac.V2.PositiveFrequencyPacketBridge

@[expose] public section

namespace LeanDirac.V4

/-!
# V4-005: analytic C19 packet-current bridge used by the localized readout

The exact finite-C19 theorem below is inherited from the native phase/current
derivation, not from the numerical comparator.  V4-003 fixes the localized
carrier readout and V4-005's artifact gate pulls the corresponding local phase
variation through that same native map.  This module records the analytic
transport/current identity used in that audit.

This certificate does not yet assert a complete cell crossing.  The final
same-carrier analytic assembly, including the controlled localized remainder,
is intentionally deferred to V4-010.
-/

open LeanDirac.V2

structure LocalizedReadoutTransportCertificate : Prop where
  packetCurrentBridge : ∀ (L : ℕ) [NeZero L] (mass : ℝ)
      (energy : PeriodicCubicMomentum L → ℝ)
      (amplitude : PeriodicCubicMomentum L → UniformCell2)
      (_henergy : ∀ mode, energy mode ≠ 0)
      (_hpositive : ∀ mode, 0 < energy mode + mass)
      (_hshell : ∀ mode,
        (energy mode)^2 = mass^2 +
          (c19ContinuumSpeed * periodicQuantizedMomentum L mode 0)^2 +
          (c19ContinuumSpeed * periodicQuantizedMomentum L mode 1)^2 +
          (c19ContinuumSpeed * periodicQuantizedMomentum L mode 2)^2)
      (axis : Fin 3),
    periodicAveragedNativePositiveAxisFlux L axis amplitude =
      periodicAveragedDiracCurrent axis
          (periodicPositiveSpectralPacket L mass energy amplitude) +
        ∑ mode, finiteC19CurrentRemainder
          (periodicQuantizedMomentum L mode) axis (amplitude mode)

theorem localized_readout_transport_current :
    LocalizedReadoutTransportCertificate where
  packetCurrentBridge :=
    periodic_integrated_positive_packet_current_bridge

end LeanDirac.V4

#print axioms LeanDirac.V4.localized_readout_transport_current
