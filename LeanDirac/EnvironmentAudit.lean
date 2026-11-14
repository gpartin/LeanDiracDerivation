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
        -- `moduleData.constants` is the serialized import interface.  In
        -- module mode it may erase an opaque theorem proof and represent the
        -- imported declaration as an axiom-shaped interface entry.  Query the
        -- reconstructed environment instead; that retains the declaration's
        -- actual kind and therefore distinguishes theorems from user axioms.
        match env.find? name with
        | some info =>
            let axioms ← Lean.collectAxioms name
            -- Opaque theorems imported through a module interface can be
            -- represented by `axiomInfo` after proof erasure.  Such a theorem
            -- does not list itself among its logical axioms; a genuine user
            -- axiom does.  This semantic test catches the latter without
            -- rejecting ordinary opaque proofs.
            if info matches .axiomInfo _ then
              if axioms.contains name then
                throwError "custom axiom declaration in production module: {name}"
            if axioms.contains ``sorryAx then
              throwError "production declaration contains sorry: {name}"
            for axiomName in axioms do
              unless allowedFoundationAxiom axiomName do
                throwError
                  "production declaration {name} depends on nonstandard axiom: {axiomName}"
        | none =>
            throwError "compiled production declaration missing from environment: {name}"
  logInfo "LEAN_DIRAC_ENVIRONMENT_AUDIT_PASS"
