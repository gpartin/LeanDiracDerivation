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


def main() -> int:
    for name, source in FIXTURES.items():
        assert_rejected(name, source)
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
