module

public import Mathlib.Util.AssertNoSorry

@[expose] public section

/-! Compiler-environment audit used by the publication boundary. -/

open Lean Meta Elab Command

private meta def allowedFoundationAxiom (name : Name) : Bool :=
  name == ``propext || name == ``Classical.choice || name == ``Quot.sound

/--
Audit every compiled declaration owned by a `LeanDirac` production module.
This operates on Lean's environment, so source-level comments, literals,
escaped identifiers and declaration modifiers cannot disguise a custom axiom,
`sorry`, or a dependency on a nonstandard axiom imported from another module.
-/
elab "#audit_lean_dirac_environment" : command => do
  let env ← getEnv
  for h : moduleIdx in *...env.header.moduleNames.size do
    let moduleName := env.header.moduleNames[moduleIdx]
    if moduleName == `LeanDirac || moduleName == `LeanDiracAudit ||
        moduleName.toString.startsWith "LeanDirac." then
      logInfo m!"LEAN_DIRAC_AUDITED_MODULE {moduleName}"
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
            for axiomName in axioms do
              unless allowedFoundationAxiom axiomName do
                throwError
                  "production declaration {name} depends on nonstandard axiom: {axiomName}"
  logInfo "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS"
