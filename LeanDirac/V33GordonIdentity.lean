import LeanDirac.V33Helicity

/-!
# V3.3 Stage 7: forward on-shell Gordon relation

This is the forward, equal-momentum current relation for the normalized free
state.  It is not a non-forward electromagnetic vertex and does not prove
`g = 2`.
-/

namespace LeanDirac

noncomputable section

theorem v33_forward_gordon_time
    (mass energy px py pz : ℝ)
    (hpositive : 0 < energy + mass)
    (hshell : energy^2 = mass^2 + px^2 + py^2 + pz^2)
    (q : UniformCell2) :
    mass * cubicalCell4Density
        (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
      energy * v33ScalarBilinear
        (normalizedPositiveFrequencyEmbedding mass energy px py pz q) := by
  rw [normalized_positive_frequency_density mass energy px py pz
    hpositive hshell q]
  rw [v33_normalized_scalar_bilinear mass energy px py pz
    hpositive hshell q]
  ring

theorem v33_forward_gordon_space
    (mass energy px py pz : ℝ)
    (hpositive : 0 < energy + mass)
    (hshell : energy^2 = mass^2 + px^2 + py^2 + pz^2)
    (q : UniformCell2) (axis : Fin 3) :
    mass * cubicalDiracCurrent axis
        (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
      c19ContinuumSpeed *
        (fun axis : Fin 3 =>
          if axis = 0 then px else if axis = 1 then py else pz) axis *
        v33ScalarBilinear
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) := by
  rw [normalized_positive_frequency_current mass energy px py pz
    hpositive hshell q axis]
  rw [v33_normalized_scalar_bilinear mass energy px py pz
    hpositive hshell q]
  ring

structure V33ForwardGordonCertificate : Prop where
  time : ∀ mass energy px py pz,
    0 < energy + mass →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 → ∀ q,
      mass * cubicalCell4Density
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
        energy * v33ScalarBilinear
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q)
  space : ∀ mass energy px py pz,
    0 < energy + mass →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 → ∀ q axis,
      mass * cubicalDiracCurrent axis
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
        c19ContinuumSpeed *
          (fun axis : Fin 3 =>
            if axis = 0 then px else if axis = 1 then py else pz) axis *
          v33ScalarBilinear
            (normalizedPositiveFrequencyEmbedding mass energy px py pz q)

theorem v33_forward_gordon : V33ForwardGordonCertificate where
  time := v33_forward_gordon_time
  space := v33_forward_gordon_space

#print axioms v33_forward_gordon

end
end LeanDirac
