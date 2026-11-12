import Lake
open Lake DSL

package «LeanDiracDerivation» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0-rc2"

@[default_target]
lean_lib LeanDirac

lean_lib LeanDiracAudit

lean_lib Challenge where
  roots := #[`Challenge]

lean_lib Solution where
  roots := #[`Solution]

