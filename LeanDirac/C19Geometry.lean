import LeanDirac.Foundation

namespace LeanDirac

/-!
DIRAC-001 substrate geometry. C19 consists of the owning center site plus
eighteen neighbors: six face neighbors and twelve edge neighbors. This module
indexes those eighteen neighbors; the center is represented by `Site` itself.
-/

abbrev C19Neighbor := Fin 18

def c19Coord (axis : Fin 3) (neighbor : C19Neighbor) : ℤ :=
  match neighbor.val, axis.val with
  | 0, 0 => 1 | 1, 0 => -1 | 2, 1 => 1 | 3, 1 => -1
  | 4, 2 => 1 | 5, 2 => -1
  | 6, 0 => -1 | 6, 1 => -1
  | 7, 0 => -1 | 7, 1 => 1
  | 8, 0 => 1 | 8, 1 => -1
  | 9, 0 => 1 | 9, 1 => 1
  | 10, 0 => -1 | 10, 2 => -1
  | 11, 0 => -1 | 11, 2 => 1
  | 12, 0 => 1 | 12, 2 => -1
  | 13, 0 => 1 | 13, 2 => 1
  | 14, 1 => -1 | 14, 2 => -1
  | 15, 1 => -1 | 15, 2 => 1
  | 16, 1 => 1 | 16, 2 => -1
  | 17, 1 => 1 | 17, 2 => 1
  | _, _ => 0

def c19Weight (neighbor : C19Neighbor) : ℝ :=
  if neighbor.val < 6 then 2 else 1

theorem c19_weight_positive (neighbor : C19Neighbor) :
    0 < c19Weight neighbor := by
  unfold c19Weight
  split <;> norm_num

structure C19Lattice (Site : Type) where
  neighbor : Site → C19Neighbor → Site

abbrev C19Edge (Site : Type) := Site × C19Neighbor

def c19Bond (geometry : C19Lattice Site) (edge : C19Edge Site) : Bond Site where
  source := edge.1
  target := geometry.neighbor edge.1 edge.2
  weight := c19Weight edge.2

theorem c19_bond_weight_positive (geometry : C19Lattice Site)
    (edge : C19Edge Site) :
    0 < (c19Bond geometry edge).weight :=
  c19_weight_positive edge.2

end LeanDirac
