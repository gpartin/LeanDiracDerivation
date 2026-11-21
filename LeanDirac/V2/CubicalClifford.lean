/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import Mathlib.Data.Complex.Basic
public import Mathlib.Tactic

@[expose] public section

namespace LeanDirac.V2

/-!
# Cubical Clifford algebra without a prepared field background

This module constructs the three spatial Clifford generators and the parity
grading from the oriented incidence geometry of one abstract cube.  It does
not import the native-field action, the prepared tetrahedral rays, or any
previous Dirac stage.  Conventional four-component matrices occur only
downstream, as coordinates on the two rank-four flavor sectors of the
eight-corner incidence representation.
-/

noncomputable section
open Complex

abbrev CubicalCorner := Fin 8
abbrev CubicalCell8 := CubicalCorner → ℂ
abbrev CubicalCell4 := Fin 4 → ℂ

/-! ## Oriented cube incidence -/

def cubicalFlip0 : CubicalCorner → CubicalCorner
  | 0 => 4 | 1 => 5 | 2 => 6 | 3 => 7
  | 4 => 0 | 5 => 1 | 6 => 2 | _ => 3

def cubicalFlip1 : CubicalCorner → CubicalCorner
  | 0 => 2 | 1 => 3 | 2 => 0 | 3 => 1
  | 4 => 6 | 5 => 7 | 6 => 4 | _ => 5

def cubicalFlip2 : CubicalCorner → CubicalCorner
  | 0 => 1 | 1 => 0 | 2 => 3 | 3 => 2
  | 4 => 5 | 5 => 4 | 6 => 7 | _ => 6

def cubicalBit0 : CubicalCorner → Nat
  | 0 | 1 | 2 | 3 => 0
  | _ => 1

def cubicalBit1 : CubicalCorner → Nat
  | 0 | 1 | 4 | 5 => 0
  | _ => 1

def cubicalBit2 : CubicalCorner → Nat
  | 0 | 2 | 4 | 6 => 0
  | _ => 1

def cubicalFlip (axis : Fin 3) : CubicalCorner → CubicalCorner :=
  match axis with
  | 0 => cubicalFlip0
  | 1 => cubicalFlip1
  | _ => cubicalFlip2

/-- Koszul incidence sign: an axis crossing counts the lower-index bits. -/
def cubicalIncidenceSign (axis : Fin 3) (corner : CubicalCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cubicalBit0 corner = 0 then 1 else -1
  | _ => if (cubicalBit0 corner + cubicalBit1 corner) % 2 = 0 then 1 else -1

def cubicalGamma (axis : Fin 3) (amplitude : CubicalCell8) : CubicalCell8 :=
  fun corner => cubicalIncidenceSign axis corner *
    amplitude (cubicalFlip axis corner)

def cubicalBeta (amplitude : CubicalCell8) : CubicalCell8 := fun corner =>
  if (cubicalBit0 corner + cubicalBit1 corner + cubicalBit2 corner) % 2 = 0 then
    amplitude corner
  else
    -amplitude corner

theorem cubical_gamma_square (axis : Fin 3) (amplitude : CubicalCell8) :
    cubicalGamma axis (cubicalGamma axis amplitude) = amplitude := by
  funext corner
  fin_cases axis <;> fin_cases corner <;>
    simp [cubicalGamma, cubicalIncidenceSign, cubicalFlip,
      cubicalFlip0, cubicalFlip1, cubicalFlip2, cubicalBit0, cubicalBit1]

theorem cubical_gamma_anticommute
    (axis other : Fin 3) (haxis : axis ≠ other) (amplitude : CubicalCell8) :
    cubicalGamma axis (cubicalGamma other amplitude) =
      fun corner => -cubicalGamma other (cubicalGamma axis amplitude) corner := by
  fin_cases axis <;> fin_cases other <;> try { contradiction }
  all_goals
    funext corner
    fin_cases corner <;>
      simp [cubicalGamma, cubicalIncidenceSign, cubicalFlip,
        cubicalFlip0, cubicalFlip1, cubicalFlip2, cubicalBit0, cubicalBit1]

theorem cubical_beta_square (amplitude : CubicalCell8) :
    cubicalBeta (cubicalBeta amplitude) = amplitude := by
  funext corner
  fin_cases corner <;>
    simp [cubicalBeta, cubicalBit0, cubicalBit1, cubicalBit2]

theorem cubical_beta_anticommutes
    (axis : Fin 3) (amplitude : CubicalCell8) :
    cubicalBeta (cubicalGamma axis amplitude) =
      fun corner => -cubicalGamma axis (cubicalBeta amplitude) corner := by
  fin_cases axis
  all_goals
    funext corner
    fin_cases corner <;>
      simp [cubicalBeta, cubicalGamma, cubicalIncidenceSign, cubicalFlip,
        cubicalFlip0, cubicalFlip1, cubicalFlip2,
        cubicalBit0, cubicalBit1, cubicalBit2]

def cubicalIncidenceLinear
    (k0 k1 k2 : ℂ) (amplitude : CubicalCell8) : CubicalCell8 := fun corner =>
  k0 * cubicalGamma 0 amplitude corner +
    k1 * cubicalGamma 1 amplitude corner +
    k2 * cubicalGamma 2 amplitude corner

theorem cubical_incidence_linear_square
    (k0 k1 k2 : ℂ) (amplitude : CubicalCell8) :
    cubicalIncidenceLinear k0 k1 k2
        (cubicalIncidenceLinear k0 k1 k2 amplitude) =
      fun corner => (k0^2 + k1^2 + k2^2) * amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [cubicalIncidenceLinear, cubicalGamma, cubicalIncidenceSign,
      cubicalFlip, cubicalFlip0, cubicalFlip1, cubicalFlip2,
      cubicalBit0, cubicalBit1] <;> ring

/-! ## The two geometry-derived rank-four flavor sectors -/

def cubicalFlavorSign (corner : CubicalCorner) : ℝ :=
  if cubicalBit2 corner % 2 = 0 then 1 else -1

def cubicalFlavor (amplitude : CubicalCell8) : CubicalCell8 := fun corner =>
  I * (cubicalFlavorSign corner : ℂ) *
    amplitude (cubicalFlip1 (cubicalFlip2 corner))

theorem cubical_flavor_involution (amplitude : CubicalCell8) :
    cubicalFlavor (cubicalFlavor amplitude) = amplitude := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavor, cubicalFlavorSign, cubicalFlip1, cubicalFlip2,
      cubicalBit2] <;> ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cubical_flavor_commutes_gamma
    (axis : Fin 3) (amplitude : CubicalCell8) :
    cubicalFlavor (cubicalGamma axis amplitude) =
      cubicalGamma axis (cubicalFlavor amplitude) := by
  funext corner
  fin_cases axis <;> fin_cases corner <;>
    simp [cubicalFlavor, cubicalFlavorSign, cubicalGamma,
      cubicalIncidenceSign, cubicalFlip, cubicalFlip0, cubicalFlip1,
      cubicalFlip2, cubicalBit0, cubicalBit1, cubicalBit2]

theorem cubical_flavor_commutes_beta (amplitude : CubicalCell8) :
    cubicalFlavor (cubicalBeta amplitude) =
      cubicalBeta (cubicalFlavor amplitude) := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavor, cubicalFlavorSign, cubicalBeta,
      cubicalFlip1, cubicalFlip2, cubicalBit0, cubicalBit1, cubicalBit2]

def cubicalEmbedMinus (z : CubicalCell4) : CubicalCell8
  | 0 => z 0 | 1 => z 1 | 2 => -I * z 1 | 3 => I * z 0
  | 4 => z 2 | 5 => z 3 | 6 => -I * z 3 | _ => I * z 2

def cubicalEmbedPlus (z : CubicalCell4) : CubicalCell8
  | 0 => z 0 | 1 => z 1 | 2 => I * z 1 | 3 => -I * z 0
  | 4 => z 2 | 5 => z 3 | 6 => I * z 3 | _ => -I * z 2

theorem cubical_embed_minus_injective : Function.Injective cubicalEmbedMinus := by
  intro first second h
  funext component
  fin_cases component
  · exact congrFun h 0
  · exact congrFun h 1
  · exact congrFun h 4
  · exact congrFun h 5

theorem cubical_embed_plus_injective : Function.Injective cubicalEmbedPlus := by
  intro first second h
  funext component
  fin_cases component
  · exact congrFun h 0
  · exact congrFun h 1
  · exact congrFun h 4
  · exact congrFun h 5

theorem cubical_embed_minus_has_negative_flavor (z : CubicalCell4) :
    cubicalFlavor (cubicalEmbedMinus z) =
      fun corner => -cubicalEmbedMinus z corner := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavor, cubicalFlavorSign, cubicalEmbedMinus,
      cubicalFlip1, cubicalFlip2, cubicalBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cubical_embed_plus_has_positive_flavor (z : CubicalCell4) :
    cubicalFlavor (cubicalEmbedPlus z) = cubicalEmbedPlus z := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavor, cubicalFlavorSign, cubicalEmbedPlus,
      cubicalFlip1, cubicalFlip2, cubicalBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

/-! ## Downstream standard four-component coordinates -/

def cubicalStandardBeta (z : CubicalCell4) : CubicalCell4
  | 0 => z 0 | 1 => z 1 | 2 => -z 2 | _ => -z 3

def cubicalStandardAlphaX (z : CubicalCell4) : CubicalCell4
  | 0 => z 3 | 1 => z 2 | 2 => z 1 | _ => z 0

def cubicalStandardAlphaY (z : CubicalCell4) : CubicalCell4
  | 0 => -I * z 3 | 1 => I * z 2 | 2 => -I * z 1 | _ => I * z 0

def cubicalStandardAlphaZ (z : CubicalCell4) : CubicalCell4
  | 0 => z 2 | 1 => -z 3 | 2 => z 0 | _ => -z 1

def cubicalStandardSpatial
    (px py pz : ℂ) (z : CubicalCell4) : CubicalCell4 := fun component =>
  px * cubicalStandardAlphaX z component +
    py * cubicalStandardAlphaY z component +
    pz * cubicalStandardAlphaZ z component

def cubicalStandardHamiltonian
    (mass px py pz : ℂ) (z : CubicalCell4) : CubicalCell4 := fun component =>
  mass * cubicalStandardBeta z component +
    cubicalStandardSpatial px py pz z component

def standardToCubicalMinus (z : CubicalCell4) : CubicalCell4
  | 0 => z 0 | 1 => -z 3 | 2 => z 2 | _ => z 1

def standardToCubicalPlus (z : CubicalCell4) : CubicalCell4
  | 0 => -z 1 | 1 => z 2 | 2 => z 3 | _ => z 0

def cubicalDiracEmbedMinus (z : CubicalCell4) : CubicalCell8 :=
  cubicalEmbedMinus (standardToCubicalMinus z)

def cubicalDiracEmbedPlus (z : CubicalCell4) : CubicalCell8 :=
  cubicalEmbedPlus (standardToCubicalPlus z)

theorem cubical_dirac_embed_minus_injective :
    Function.Injective cubicalDiracEmbedMinus := by
  intro first second h
  have hcube := cubical_embed_minus_injective h
  funext component
  fin_cases component
  · exact congrFun hcube 0
  · exact congrFun hcube 3
  · exact congrFun hcube 2
  · have := congrFun hcube 1
    simpa [standardToCubicalMinus] using congrArg Neg.neg this

theorem cubical_dirac_embed_plus_injective :
    Function.Injective cubicalDiracEmbedPlus := by
  intro first second h
  have hcube := cubical_embed_plus_injective h
  funext component
  fin_cases component
  · exact congrFun hcube 3
  · have := congrFun hcube 0
    simpa [standardToCubicalPlus] using congrArg Neg.neg this
  · exact congrFun hcube 1
  · exact congrFun hcube 2

def cubicalNativeSpatial
    (px py pz : ℂ) (amplitude : CubicalCell8) : CubicalCell8 :=
  cubicalIncidenceLinear pz (-py) (-px) amplitude

def cubicalNativeHamiltonian
    (mass px py pz : ℂ) (amplitude : CubicalCell8) : CubicalCell8 :=
  fun corner => mass * cubicalBeta amplitude corner +
    cubicalNativeSpatial px py pz amplitude corner

theorem cubical_beta_intertwines_minus (z : CubicalCell4) :
    cubicalBeta (cubicalDiracEmbedMinus z) =
      cubicalDiracEmbedMinus (cubicalStandardBeta z) := by
  funext corner
  fin_cases corner <;>
    simp [cubicalBeta, cubicalDiracEmbedMinus, cubicalEmbedMinus,
      standardToCubicalMinus, cubicalStandardBeta,
      cubicalBit0, cubicalBit1, cubicalBit2]

theorem cubical_beta_intertwines_plus (z : CubicalCell4) :
    cubicalBeta (cubicalDiracEmbedPlus z) =
      cubicalDiracEmbedPlus (cubicalStandardBeta z) := by
  funext corner
  fin_cases corner <;>
    simp [cubicalBeta, cubicalDiracEmbedPlus, cubicalEmbedPlus,
      standardToCubicalPlus, cubicalStandardBeta,
      cubicalBit0, cubicalBit1, cubicalBit2]

theorem cubical_native_spatial_intertwines_minus
    (px py pz : ℂ) (z : CubicalCell4) :
    cubicalNativeSpatial px py pz (cubicalDiracEmbedMinus z) =
      cubicalDiracEmbedMinus (cubicalStandardSpatial px py pz z) := by
  funext corner
  fin_cases corner <;>
    simp [cubicalNativeSpatial, cubicalIncidenceLinear, cubicalGamma,
      cubicalIncidenceSign, cubicalFlip, cubicalFlip0, cubicalFlip1,
      cubicalFlip2, cubicalDiracEmbedMinus, cubicalEmbedMinus,
      standardToCubicalMinus, cubicalStandardSpatial,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ, cubicalBit0, cubicalBit1] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cubical_native_spatial_intertwines_plus
    (px py pz : ℂ) (z : CubicalCell4) :
    cubicalNativeSpatial px py pz (cubicalDiracEmbedPlus z) =
      cubicalDiracEmbedPlus (cubicalStandardSpatial px py pz z) := by
  funext corner
  fin_cases corner <;>
    simp [cubicalNativeSpatial, cubicalIncidenceLinear, cubicalGamma,
      cubicalIncidenceSign, cubicalFlip, cubicalFlip0, cubicalFlip1,
      cubicalFlip2, cubicalDiracEmbedPlus, cubicalEmbedPlus,
      standardToCubicalPlus, cubicalStandardSpatial,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ, cubicalBit0, cubicalBit1] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cubical_hamiltonian_intertwines_minus
    (mass px py pz : ℂ) (z : CubicalCell4) :
    cubicalNativeHamiltonian mass px py pz (cubicalDiracEmbedMinus z) =
      cubicalDiracEmbedMinus
        (cubicalStandardHamiltonian mass px py pz z) := by
  funext corner
  rw [show cubicalNativeHamiltonian mass px py pz
      (cubicalDiracEmbedMinus z) corner =
      mass * cubicalBeta (cubicalDiracEmbedMinus z) corner +
        cubicalNativeSpatial px py pz (cubicalDiracEmbedMinus z) corner by rfl]
  rw [congrFun (cubical_beta_intertwines_minus z) corner,
    congrFun (cubical_native_spatial_intertwines_minus px py pz z) corner]
  fin_cases corner <;>
    simp [cubicalDiracEmbedMinus, cubicalEmbedMinus,
      standardToCubicalMinus, cubicalStandardHamiltonian,
      cubicalStandardBeta, cubicalStandardSpatial,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ] <;> ring

theorem cubical_hamiltonian_intertwines_plus
    (mass px py pz : ℂ) (z : CubicalCell4) :
    cubicalNativeHamiltonian mass px py pz (cubicalDiracEmbedPlus z) =
      cubicalDiracEmbedPlus
        (cubicalStandardHamiltonian mass px py pz z) := by
  funext corner
  rw [show cubicalNativeHamiltonian mass px py pz
      (cubicalDiracEmbedPlus z) corner =
      mass * cubicalBeta (cubicalDiracEmbedPlus z) corner +
        cubicalNativeSpatial px py pz (cubicalDiracEmbedPlus z) corner by rfl]
  rw [congrFun (cubical_beta_intertwines_plus z) corner,
    congrFun (cubical_native_spatial_intertwines_plus px py pz z) corner]
  fin_cases corner <;>
    simp [cubicalDiracEmbedPlus, cubicalEmbedPlus,
      standardToCubicalPlus, cubicalStandardHamiltonian,
      cubicalStandardBeta, cubicalStandardSpatial,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ] <;> ring

theorem cubical_standard_hamiltonian_square
    (mass px py pz : ℂ) (z : CubicalCell4) :
    cubicalStandardHamiltonian mass px py pz
        (cubicalStandardHamiltonian mass px py pz z) =
      fun component => (mass^2 + px^2 + py^2 + pz^2) * z component := by
  funext component
  fin_cases component <;>
    simp [cubicalStandardHamiltonian, cubicalStandardBeta,
      cubicalStandardSpatial, cubicalStandardAlphaX,
      cubicalStandardAlphaY, cubicalStandardAlphaZ] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

end
end LeanDirac.V2

#print axioms LeanDirac.V2.cubical_gamma_anticommute
#print axioms LeanDirac.V2.cubical_hamiltonian_intertwines_minus
#print axioms LeanDirac.V2.cubical_hamiltonian_intertwines_plus
