"""Exercise dependency-boundary bypasses against Lean's own parser."""

from __future__ import annotations

from pathlib import Path

import production_guard as guard


FIXTURES = {
    "comment_hidden": (
        "import LeanDirac.Publication\n\n"
        "/- hidden after the apparent header -/ import LeanDirac.Foundation\n"
    ),
    "tab_separated": "import\tLeanDirac.Publication\n\n",
    "multiple_modules": (
        "import LeanDirac.Publication LeanDirac.Foundation\n\n"
    ),
}


def assert_rejected(name: str, source: str) -> None:
    path = guard.ROOT / f"GuardBypass_{name}.lean"
    path.write_text(source, encoding="utf-8", newline="\n")
    try:
        errors: list[str] = []
        guard.require_imports(
            path, ["LeanDirac.Publication"], errors, f"fixture {name}"
        )
        if not errors:
            raise AssertionError(f"dependency bypass was accepted: {name}")
    finally:
        path.unlink(missing_ok=True)


def assert_dependency_output_rejected(name: str, output: str) -> None:
    _, errors = guard.decode_lean_dependency_output(output)
    if not errors:
        raise AssertionError(f"dependency-path bypass was accepted: {name}")


def assert_relative_approved_dependency() -> None:
    modules, errors = guard.decode_lean_dependency_output(
        ".lake/build/lib/lean/LeanDirac/Publication.olean\n"
    )
    if errors or modules != ["LeanDirac.Publication"]:
        raise AssertionError(
            f"valid ROOT-relative dependency rejected: modules={modules}, errors={errors}"
        )


def run_lean_fixture(name: str, source: str):
    path = guard.ROOT / f"GuardSemantic_{name}.lean"
    path.write_text(source, encoding="utf-8", newline="\n")
    try:
        return guard.subprocess.run(
            ["lake", "env", "lean", str(path.relative_to(guard.ROOT))],
            cwd=guard.ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_lean_accepted(name: str, source: str) -> None:
    """Prove that the fixture toolchain reaches and accepts a valid contract."""
    completed = run_lean_fixture(name, source)
    if completed.returncode != 0:
        raise AssertionError(
            f"known-good Lean contract failed before negative tests: {name}\n"
            + completed.stdout
            + completed.stderr
        )


def assert_lean_rejected_at_rfl(name: str, source: str) -> None:
    """Require failure at the intended definitional-equality proof."""
    completed = run_lean_fixture(name, source)
    output = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise AssertionError(f"circular stationarity contract was accepted: {name}")
    expected = "Tactic `rfl` failed"
    if expected not in output:
        raise AssertionError(
            f"negative Lean fixture failed for an unrelated reason: {name}\n{output}"
        )


def main() -> int:
    for name, source in FIXTURES.items():
        assert_rejected(name, source)
    assert_dependency_output_rejected(
        "multiple_lib_lean_markers",
        "/tmp/lib/lean/Unapproved/lib/lean/LeanDirac/Publication.olean\n",
    )
    assert_dependency_output_rejected(
        "same_module_outside_approved_root",
        "/tmp/foreign/lib/lean/LeanDirac/Publication.olean\n",
    )
    assert_dependency_output_rejected(
        "fake_init_outside_toolchain", "/tmp/foreign/lib/lean/Foo/Init.olean\n"
    )
    approved = (
        guard.ROOT
        / ".lake"
        / "build"
        / "lib"
        / "lean"
        / "LeanDirac"
        / "Publication.olean"
    ).resolve()
    assert_dependency_output_rejected(
        "duplicate_approved_module", f"{approved}\n{approved}\n"
    )
    assert_relative_approved_dependency()
    assert_lean_accepted(
        "known_good_contract",
        "import Mathlib\n\n"
        "def Expected (action : Prop) : Prop := action\n"
        "def Candidate (action : Prop) : Prop := action\n"
        "example (action : Prop) : "
        "Candidate action ↔ Expected action := by rfl\n",
    )
    assert_lean_rejected_at_rfl(
        "helper_alias",
        "import Mathlib\n\n"
        "def Expected (action : Prop) : Prop := action\n"
        "def HiddenTarget (target : Prop) : Prop := target\n"
        "def Candidate (action target : Prop) : Prop := "
        "action ∧ HiddenTarget target\n"
        "example (action target : Prop) : "
        "Candidate action target ↔ Expected action := by rfl\n",
    )
    assert_lean_rejected_at_rfl(
        "expanded_target",
        "import Mathlib\n\n"
        "def Expected (action : Prop) : Prop := action\n"
        "def Candidate (action target : Prop) : Prop := action ∧ target\n"
        "example (action target : Prop) : "
        "Candidate action target ↔ Expected action := by rfl\n",
    )
    assert_lean_rejected_at_rfl(
        "derivative_alias_redefinition",
        "import Mathlib\n\n"
        "def ExpectedDerivative (f : ℝ → ℝ) : Prop := HasDerivAt f 0 0\n"
        "def ProductionDerivative (_f : ℝ → ℝ) : Prop := True\n"
        "example (f : ℝ → ℝ) : "
        "ProductionDerivative f ↔ ExpectedDerivative f := by rfl\n",
    )
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
