import LeanDirac.Foundation

namespace LeanDirac

/-! DIRAC-001 substrate geometry: six face and twelve edge neighbors. -/

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

end LeanDirac
