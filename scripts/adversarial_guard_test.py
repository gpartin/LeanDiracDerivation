"""Small negative controls for the manual production audit."""

from __future__ import annotations

import production_guard as guard


def main() -> int:
    for token in ("by sorry", "by admit", "axiom injected : Prop"):
        if not guard.forbidden_tokens(token):
            raise AssertionError(f"forbidden token escaped detection: {token}")
    if guard.forbidden_tokens("theorem clean : True := by trivial"):
        raise AssertionError("clean Lean source was rejected")
    axioms = guard.parse_axioms(
        "'x' depends on axioms: [propext, Classical.choice, Quot.sound]"
    )
    if axioms != guard.ALLOWED_AXIOMS:
        raise AssertionError(f"axiom parser mismatch: {axioms}")
    bad = guard.parse_axioms("'x' depends on axioms: [propext, sorryAx]")
    if "sorryAx" not in bad:
        raise AssertionError("sorryAx escaped the axiom parser")
    payload = guard.load(guard.STABILITY)
    if guard.validate_stability(payload):
        raise AssertionError("passing stability evidence was rejected")
    broken = dict(payload)
    broken["verdict"] = "FAIL"
    if not guard.validate_stability(broken):
        raise AssertionError("failed stability evidence was accepted")
    print("ADVERSARIAL_GUARD_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
