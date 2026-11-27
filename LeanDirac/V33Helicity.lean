import LeanDirac.V33BilinearObservables

/-!
# V3.3 Stage 6: cube-derived spin spectrum and scoped helicity

In the cubical orientation convention, `standardCubicalSpin 0` is paired with
the `pz` momentum coordinate.  This stage constructs the two axis-aligned
positive-frequency moving spinors and proves their half-valued helicities.  It
does not add a new spin postulate.
-/

namespace LeanDirac

noncomputable section

def v33Axis0Helicity (z : CubicalCell4) : CubicalCell4 :=
  standardCubicalSpin 0 z

def v33TwoChannelDown : UniformCell2
  | 0 => 1
  | _ => 0

def v33TwoChannelUp : UniformCell2
  | 0 => 0
  | _ => 1

theorem v33_axis0_helicity_down :
    v33Axis0Helicity (standardCell4Basis 0) =
      fun component => (-1 / 2 : ℂ) * standardCell4Basis 0 component := by
  exact standard_spin0_basis_spectrum.1

theorem v33_axis0_helicity_up :
    v33Axis0Helicity (standardCell4Basis 1) =
      fun component => (1 / 2 : ℂ) * standardCell4Basis 1 component := by
  exact standard_spin0_basis_spectrum.2.1

theorem v33_axis0_helicity_square (z : CubicalCell4) :
    v33Axis0Helicity (v33Axis0Helicity z) =
      fun component => (1 / 4 : ℂ) * z component := by
  exact standard_spin_component_square 0 z

theorem v33_axis_aligned_moving_helicity_down
    (mass energy pz : ℝ) :
    v33Axis0Helicity
        (normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelDown) =
      fun component => (-1 / 2 : ℂ) *
        normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelDown component := by
  funext component
  fin_cases component <;>
    simp [v33Axis0Helicity, v33TwoChannelDown,
      normalizedPositiveFrequencyEmbedding, positivePauliSpatial,
      standardCubicalSpin, standardCubicalPlaneBivector,
      standardCubicalGamma, cubicalStandardAlphaX,
      cubicalStandardAlphaY] <;>
    ring_nf <;> norm_num [Complex.I_sq]

theorem v33_axis_aligned_moving_helicity_up
    (mass energy pz : ℝ) :
    v33Axis0Helicity
        (normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelUp) =
      fun component => (1 / 2 : ℂ) *
        normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelUp component := by
  funext component
  fin_cases component <;>
    simp [v33Axis0Helicity, v33TwoChannelUp,
      normalizedPositiveFrequencyEmbedding, positivePauliSpatial,
      standardCubicalSpin, standardCubicalPlaneBivector,
      standardCubicalGamma, cubicalStandardAlphaX,
      cubicalStandardAlphaY] <;>
    ring_nf <;> norm_num [Complex.I_sq]

structure V33HelicityCertificate : Prop where
  down : v33Axis0Helicity (standardCell4Basis 0) =
    fun component => (-1 / 2 : ℂ) * standardCell4Basis 0 component
  up : v33Axis0Helicity (standardCell4Basis 1) =
    fun component => (1 / 2 : ℂ) * standardCell4Basis 1 component
  square : ∀ z, v33Axis0Helicity (v33Axis0Helicity z) =
    fun component => (1 / 4 : ℂ) * z component
  movingDown : ∀ mass energy pz,
    v33Axis0Helicity
        (normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelDown) =
      fun component => (-1 / 2 : ℂ) *
        normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelDown component
  movingUp : ∀ mass energy pz,
    v33Axis0Helicity
        (normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelUp) =
      fun component => (1 / 2 : ℂ) *
        normalizedPositiveFrequencyEmbedding
          mass energy 0 0 pz v33TwoChannelUp component

theorem v33_helicity : V33HelicityCertificate where
  down := v33_axis0_helicity_down
  up := v33_axis0_helicity_up
  square := v33_axis0_helicity_square
  movingDown := v33_axis_aligned_moving_helicity_down
  movingUp := v33_axis_aligned_moving_helicity_up

#print axioms v33_helicity

end
end LeanDirac
