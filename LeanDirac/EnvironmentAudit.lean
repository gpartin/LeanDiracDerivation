import Mathlib.Util.AssertNoSorry

/-! Compiler-environment audit used by the publication boundary. -/

open Lean Meta Elab Command

/--
Audit every compiled declaration owned by a `LeanDirac` module. This operates
on Lean's environment, so source-level comments, literals, escaped identifiers
and declaration modifiers cannot disguise a custom axiom or sorry.
-/
elab "#audit_lean_dirac_environment" : command => do
  let env ← getEnv
  for h : moduleIdx in *...env.header.moduleNames.size do
    let moduleName := env.header.moduleNames[moduleIdx]
    if moduleName == `LeanDirac || moduleName.toString.startsWith "LeanDirac." then
      let moduleData := env.header.moduleData[moduleIdx]!
      for h : declarationIdx in *...moduleData.constNames.size do
        let name := moduleData.constNames[declarationIdx]
        let info := moduleData.constants[declarationIdx]!
        match info with
        | .axiomInfo _ =>
            throwError "custom axiom declaration in production module: {name}"
        | _ =>
            let axioms ← Lean.collectAxioms name
            if axioms.contains ``sorryAx then
              throwError "production declaration contains sorry: {name}"
  logInfo "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS"
