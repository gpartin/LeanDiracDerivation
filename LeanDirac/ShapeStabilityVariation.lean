import LeanDirac.Dirac003
import LeanDirac.ShapeStability

namespace LeanDirac

/-!
Polynomial certificate for the exact state variation of the covariance shape
energy.  Encoding the line as an explicit real polynomial prevents the proof
from assuming the force it is meant to derive.
-/

noncomputable section
open BigOperators Complex Polynomial

abbrev ComplexPolynomial := Polynomial ℝ × Polynomial ℝ

def complexPolynomialEval (p : ComplexPolynomial) (x : ℝ) : ℂ :=
  ⟨p.1.eval x, p.2.eval x⟩

def complexPolynomialConst (z : ℂ) : ComplexPolynomial :=
  (C z.re, C z.im)

def complexPolynomialLine (z dz : ℂ) : ComplexPolynomial :=
  (C z.re + C dz.re * X, C z.im + C dz.im * X)

def complexPolynomialAdd (p q : ComplexPolynomial) : ComplexPolynomial :=
  (p.1 + q.1, p.2 + q.2)

def complexPolynomialSub (p q : ComplexPolynomial) : ComplexPolynomial :=
  (p.1 - q.1, p.2 - q.2)

def complexPolynomialMul (p q : ComplexPolynomial) : ComplexPolynomial :=
  (p.1 * q.1 - p.2 * q.2, p.1 * q.2 + p.2 * q.1)

def complexPolynomialStar (p : ComplexPolynomial) : ComplexPolynomial :=
  (p.1, -p.2)

def complexPolynomialScaleReal (a : ℝ) (p : ComplexPolynomial) :
    ComplexPolynomial :=
  (C a * p.1, C a * p.2)

@[simp] theorem complex_polynomial_eval_const (z : ℂ) (x : ℝ) :
    complexPolynomialEval (complexPolynomialConst z) x = z := by
  apply Complex.ext <;> simp [complexPolynomialEval, complexPolynomialConst]

@[simp] theorem complex_polynomial_eval_line (z dz : ℂ) (x : ℝ) :
    complexPolynomialEval (complexPolynomialLine z dz) x = z + (x : ℂ) * dz := by
  apply Complex.ext <;>
    simp [complexPolynomialEval, complexPolynomialLine] <;> ring

@[simp] theorem complex_polynomial_eval_add (p q : ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (complexPolynomialAdd p q) x =
      complexPolynomialEval p x + complexPolynomialEval q x := by
  apply Complex.ext <;> simp [complexPolynomialEval, complexPolynomialAdd]

@[simp] theorem complex_polynomial_eval_native_add
    (p q : ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (p + q) x =
      complexPolynomialEval p x + complexPolynomialEval q x := by
  change complexPolynomialEval (complexPolynomialAdd p q) x = _
  exact complex_polynomial_eval_add p q x

@[simp] theorem complex_polynomial_eval_sub (p q : ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (complexPolynomialSub p q) x =
      complexPolynomialEval p x - complexPolynomialEval q x := by
  apply Complex.ext <;> simp [complexPolynomialEval, complexPolynomialSub]

@[simp] theorem complex_polynomial_eval_mul (p q : ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (complexPolynomialMul p q) x =
      complexPolynomialEval p x * complexPolynomialEval q x := by
  apply Complex.ext <;>
    simp [complexPolynomialEval, complexPolynomialMul]

@[simp] theorem complex_polynomial_eval_star (p : ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (complexPolynomialStar p) x =
      star (complexPolynomialEval p x) := by
  apply Complex.ext <;> simp [complexPolynomialEval, complexPolynomialStar]

@[simp] theorem complex_polynomial_eval_scale_real
    (a x : ℝ) (p : ComplexPolynomial) :
    complexPolynomialEval (complexPolynomialScaleReal a p) x =
      (a : ℂ) * complexPolynomialEval p x := by
  apply Complex.ext <;>
    simp [complexPolynomialEval, complexPolynomialScaleReal]

@[simp] theorem complex_polynomial_eval_sum [Fintype α]
    (f : α → ComplexPolynomial) (x : ℝ) :
    complexPolynomialEval (∑ a, f a) x =
      ∑ a, complexPolynomialEval (f a) x := by
  classical
  let s : Finset α := Finset.univ
  change complexPolynomialEval (∑ a ∈ s, f a) x =
    ∑ a ∈ s, complexPolynomialEval (f a) x
  induction s using Finset.induction_on with
  | empty =>
      apply Complex.ext <;> simp [complexPolynomialEval]
  | @insert a s ha ih =>
      rw [Finset.sum_insert ha, Finset.sum_insert ha,
        complex_polynomial_eval_native_add, ih]

def chiLinePolynomial (state direction : State Site)
    (site : Site) (component : Fin 3) : ComplexPolynomial :=
  complexPolynomialLine (state.chi site component)
    (direction.chi site component)

def chiInnerLinePolynomial (state direction : State Site)
    (left right : Site) : ComplexPolynomial :=
  ∑ component,
    complexPolynomialMul
      (complexPolynomialStar (chiLinePolynomial state direction left component))
      (chiLinePolynomial state direction right component)

def chiCovarianceActionLinePolynomial [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) (component : Fin 3) :
    ComplexPolynomial :=
  ∑ neighbor : C19Neighbor,
    complexPolynomialScaleReal (c19Weight neighbor / (8 * chi0^2))
      (complexPolynomialMul
        (chiLinePolynomial state direction
          (geometry.neighbor site neighbor) component)
        (chiInnerLinePolynomial state direction
          (geometry.neighbor site neighbor) site))

def chiCovarianceResidualLinePolynomial [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) (component : Fin 3) :
    ComplexPolynomial :=
  let chi := fun c => chiLinePolynomial state direction site c
  let cov := fun c => chiCovarianceActionLinePolynomial
    chi0 state direction geometry site c
  match component with
  | 0 => complexPolynomialSub
      (complexPolynomialMul (chi 0) (cov 1))
      (complexPolynomialMul (chi 1) (cov 0))
  | 1 => complexPolynomialSub
      (complexPolynomialMul (chi 0) (cov 2))
      (complexPolynomialMul (chi 2) (cov 0))
  | _ => complexPolynomialSub
      (complexPolynomialMul (chi 1) (cov 2))
      (complexPolynomialMul (chi 2) (cov 1))

def chiCovarianceShapeLinePolynomial [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) : Polynomial ℝ :=
  ∑ site, C (1 / 2 : ℝ) * (∑ component,
    let p := chiCovarianceResidualLinePolynomial
      chi0 state direction geometry site component
    (p.1^2 + p.2^2))

theorem chi_line_polynomial_eval (state direction : State Site)
    (site : Site) (component : Fin 3) (eps : ℝ) :
    complexPolynomialEval
        (chiLinePolynomial state direction site component) eps =
      (stateLine state direction eps).chi site component := by
  simp [chiLinePolynomial, stateLine, c3Line, c3Add, c3ScaleReal]

theorem chi_inner_line_polynomial_eval (state direction : State Site)
    (left right : Site) (eps : ℝ) :
    complexPolynomialEval
        (chiInnerLinePolynomial state direction left right) eps =
      inner ((stateLine state direction eps).chi left)
        ((stateLine state direction eps).chi right) := by
  simp [chiInnerLinePolynomial, inner, chi_line_polynomial_eval]

theorem chi_covariance_action_line_polynomial_eval [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) (component : Fin 3)
    (eps : ℝ) :
    complexPolynomialEval
        (chiCovarianceActionLinePolynomial
          chi0 state direction geometry site component) eps =
      chiCovarianceAction chi0 (stateLine state direction eps)
        geometry site component := by
  simp [chiCovarianceActionLinePolynomial, chiCovarianceAction,
    chi_line_polynomial_eval, chi_inner_line_polynomial_eval]
  apply Finset.sum_congr rfl
  intro neighbor _
  ring

theorem chi_covariance_residual_line_polynomial_eval [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) (component : Fin 3)
    (eps : ℝ) :
    complexPolynomialEval
        (chiCovarianceResidualLinePolynomial
          chi0 state direction geometry site component) eps =
      chiCovarianceShapeResidual chi0 (stateLine state direction eps)
        geometry site component := by
  fin_cases component <;>
    simp [chiCovarianceResidualLinePolynomial,
      chiCovarianceShapeResidual, wedge,
      chi_line_polynomial_eval,
      chi_covariance_action_line_polynomial_eval]

theorem chi_covariance_shape_line_polynomial_eval [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (eps : ℝ) :
    (chiCovarianceShapeLinePolynomial
        chi0 state direction geometry).eval eps =
      chiCovarianceShapeEnergy chi0 (stateLine state direction eps) geometry := by
  unfold chiCovarianceShapeLinePolynomial chiCovarianceShapeEnergy
    chiCovarianceShapeSiteEnergy c3EuclideanSq
  rw [Polynomial.eval_finsetSum]
  apply Finset.sum_congr rfl
  intro site _
  simp only [Polynomial.eval_mul, Polynomial.eval_C]
  rw [Polynomial.eval_finsetSum]
  rw [div_eq_mul_inv, mul_comm]
  simp only [one_mul]
  congr 1
  apply Finset.sum_congr rfl
  intro component _
  let p := chiCovarianceResidualLinePolynomial
    chi0 state direction geometry site component
  have hp := chi_covariance_residual_line_polynomial_eval
    chi0 state direction geometry site component eps
  have hpre := congrArg Complex.re hp
  have hpim := congrArg Complex.im hp
  change (p.1 ^ 2 + p.2 ^ 2).eval eps = _
  simp only [eval_add, eval_pow]
  change (complexPolynomialEval p eps).re ^ 2 +
    (complexPolynomialEval p eps).im ^ 2 = _
  rw [hp]
  simp [Complex.reCLM_apply, Complex.imCLM_apply]

def chiCovarianceShapeFirstVariation [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) : ℝ :=
  (chiCovarianceShapeLinePolynomial
    chi0 state direction geometry).derivative.eval 0

theorem chi_covariance_shape_hasDerivAt [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => chiCovarianceShapeEnergy chi0
        (stateLine state direction eps) geometry)
      (chiCovarianceShapeFirstVariation chi0 state direction geometry) 0 := by
  have hfun :
      (fun eps : ℝ => chiCovarianceShapeEnergy chi0
        (stateLine state direction eps) geometry) =
      (fun eps : ℝ => (chiCovarianceShapeLinePolynomial
        chi0 state direction geometry).eval eps) := by
    funext eps
    exact (chi_covariance_shape_line_polynomial_eval
      chi0 state direction geometry eps).symm
  rw [hfun]
  exact Polynomial.hasDerivAt _ _

end
end LeanDirac
