"""Fail closed if the admitted DIRAC-001 through DIRAC-006 boundary drifts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "project" / "action_lock.json"
STATUS = ROOT / "project" / "status.json"
REVIEW = ROOT / "project" / "no_bolton_review.json"

EXPECTED_GATE_MODULES = {
    "DIRAC-001": "LeanDirac/Dirac001.lean",
    "DIRAC-002": "LeanDirac/Dirac002.lean",
    "DIRAC-003": "LeanDirac/Dirac003.lean",
    "DIRAC-004": "LeanDirac/Dirac004DiscreteAction.lean",
    "DIRAC-005": "LeanDirac/Dirac005.lean",
    "DIRAC-006": "LeanDirac/Dirac006.lean",
}

EXPECTED_GATES = {
    **{f"DIRAC-{index:03d}": "PASS" for index in range(1, 7)},
    "DIRAC-007": "NEXT",
    **{f"DIRAC-{index:03d}": "OPEN" for index in range(8, 16)},
}

EXPECTED_LOCKED_FILES = {
    "LeanDirac/Foundation.lean",
    "LeanDirac/C19Geometry.lean",
    "LeanDirac/CubeGeometry.lean",
    "LeanDirac/ExteriorArea.lean",
    "LeanDirac/NativeAction.lean",
    "LeanDirac/PreparedBackground.lean",
    "LeanDirac/Dirac001.lean",
    "LeanDirac/Dirac002.lean",
    "LeanDirac/Dirac003.lean",
    "LeanDirac/Dirac003Contract.lean",
    "LeanDirac/Dirac004.lean",
    "LeanDirac/Dirac004DiscreteAction.lean",
    "LeanDirac/Dirac005.lean",
    "LeanDirac/Dirac006.lean",
    "LeanDirac/Publication.lean",
    "LeanDirac.lean",
    "LeanDiracAudit.lean",
}

AUDITED_THEOREMS = {
    "LeanDirac.dirac001_native_framework",
    "LeanDirac.periodic_c19_neighbor_reverse",
    "LeanDirac.c19_reverse_bond_target",
    "LeanDirac.dirac002_positive_periodic_c19_two_field_action",
    "LeanDirac.dirac002_prepared_geometry",
    "LeanDirac.cubic_unit_area_forces_wedge",
    "LeanDirac.psi_force_is_full_action_variation",
    "LeanDirac.chi_force_is_full_action_variation",
    "LeanDirac.psi_momentum_is_full_action_velocity_variation",
    "LeanDirac.chi_momentum_is_full_action_velocity_variation",
    "LeanDirac.dirac003_c19_local_variations_from_actual_action",
    "LeanDirac.psi_state_curve_derivative_is_native_force",
    "LeanDirac.chi_state_curve_derivative_is_native_force",
    "LeanDirac.psi_velocity_curve_derivative_is_native_momentum",
    "LeanDirac.chi_velocity_curve_derivative_is_native_momentum",
    "LeanDirac.psi_acceleration_real_coordinate",
    "LeanDirac.chi_acceleration_real_coordinate",
    "LeanDirac.native_next_is_reversible",
    "LeanDirac.centered_difference_native_next",
    "LeanDirac.dirac004_c19_reversible_action_update",
    "LeanDirac.discrete_action_hasDerivAt",
    "LeanDirac.psi_acceleration_all_directions",
    "LeanDirac.chi_acceleration_all_directions",
    "LeanDirac.native_next_is_discrete_action_stationary",
    "LeanDirac.discrete_stationarity_determines_native_next",
    "LeanDirac.dirac004_c19_update_iff_discrete_action_stationary",
    "LeanDirac.aligned_psi_cross_block_zero",
    "LeanDirac.aligned_chi_cross_block_zero",
    "LeanDirac.native_next_preserves_eight_site_sector",
    "LeanDirac.native_unit_cell_coordinate_count",
    "LeanDirac.cube_corner_coordinate_injective",
    "LeanDirac.cube_c19_neighbor_reverse",
    "LeanDirac.cube_c19_neighbor_is_periodic_c19",
    "LeanDirac.c19_direction_has_declared_class",
    "LeanDirac.dirac005_c19_native_sector",
    "LeanDirac.current_highest_gate_is_dirac005",
    "LeanDirac.aligned_area_is_product_amplitude",
    "LeanDirac.aligned_exterior_area_is_restricted_bond",
    "LeanDirac.cubic_character_neighbor",
    "LeanDirac.cubic_character_normSq_one",
    "LeanDirac.c19_weight_is_integer_weight",
    "LeanDirac.c19_first_moment_zero",
    "LeanDirac.c19_second_moment_isotropic",
    "LeanDirac.c19_transport_symbol_nonnegative",
    "LeanDirac.c19_transport_symbol_zero",
    "LeanDirac.restricted_character_energy_has_c19_symbol",
    "LeanDirac.repeated_character_bond_factorization",
    "LeanDirac.repeated_character_energy_has_c19_symbol",
    "LeanDirac.product_amplitude_first_order_expansion",
    "LeanDirac.dirac006_native_transport_symbol",
    "LeanDirac.current_highest_gate_is_dirac006",
}

ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    # Git stores these proof sources with LF endings. Normalize a Windows
    # checkout before hashing so the lock identifies repository content,
    # not the host platform's checkout convention.
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def canonical_import_modules(path: Path) -> tuple[list[str], list[str]]:
    """Parse the deliberately strict one-module-per-line import header.

    Production modules may use only `import Module.Name` with one ASCII space.
    Any other line beginning with the `import` token is rejected, including
    tabs, multiple modules, and attempted continuation syntax.
    """
    modules: list[str] = []
    errors: list[str] = []
    relative = path.relative_to(ROOT)
    lines = path.read_text(encoding="utf-8").splitlines()
    valid = re.compile(r"^import ([A-Za-z_][A-Za-z0-9_.]*)$")
    import_token = re.compile(r"^\s*import\b")

    index = 0
    while index < len(lines):
        match = valid.fullmatch(lines[index])
        if match is None:
            break
        modules.append(match.group(1))
        index += 1

    # A blank line terminates the complete import header. Requiring it makes
    # a newline-continuation such as `import Allowed\n  Unapproved` fail closed.
    if index >= len(lines) or lines[index] != "":
        errors.append(f"noncanonical import header termination: {relative}:{index + 1}")

    for number, line in enumerate(lines, 1):
        if import_token.match(line) and valid.fullmatch(line) is None:
            errors.append(f"noncanonical import syntax: {relative}:{number}")
        if number > index + 1 and import_token.match(line):
            errors.append(f"import outside canonical header: {relative}:{number}")
    return modules, errors


@lru_cache(maxsize=1)
def approved_init_dependency() -> Path:
    completed = subprocess.run(
        ["lake", "env", "lean", "--print-prefix"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    prefix = Path(completed.stdout.strip()).resolve()
    return (prefix / "lib" / "lean" / "Init.olean").resolve()


def decode_lean_dependency_output(output: str) -> tuple[list[str], list[str]]:
    """Decode only modules from this repository's exact Lean build root."""
    modules: list[str] = []
    errors: list[str] = []
    approved_root = (ROOT / ".lake" / "build" / "lib" / "lean").resolve()
    for raw in output.splitlines():
        candidate = raw.strip()
        if not candidate.endswith(".olean"):
            continue
        dependency_path = Path(candidate)
        dependency = (
            dependency_path
            if dependency_path.is_absolute()
            else ROOT / dependency_path
        ).resolve()
        if dependency == approved_init_dependency():
            continue
        try:
            relative = dependency.relative_to(approved_root)
        except ValueError:
            errors.append(f"dependency outside approved Lean build root: {raw}")
            continue
        module = ".".join(relative.with_suffix("").parts)
        modules.append(module)
    duplicates = sorted({module for module in modules if modules.count(module) > 1})
    if duplicates:
        errors.append(f"duplicate direct Lean dependencies: {duplicates}")
    return sorted(modules), errors


def lean_direct_imports(path: Path) -> tuple[list[str], list[str]]:
    """Ask Lean itself which direct modules the source imports.

    This is the security boundary. It follows Lean whitespace and comment
    syntax, so blank lines, block comments, tabs, and continued module lists
    cannot disagree with the guard's interpretation.
    """
    completed = subprocess.run(
        ["lake", "env", "lean", "--deps", str(path.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return [], [
            f"Lean dependency parse failed: {path.relative_to(ROOT)}:\n"
            + completed.stdout
            + completed.stderr
        ]
    return decode_lean_dependency_output(completed.stdout)


def require_imports(path: Path, expected: list[str], errors: list[str], label: str) -> None:
    modules, syntax_errors = canonical_import_modules(path)
    errors.extend(syntax_errors)
    if modules != expected:
        errors.append(f"{label} has noncanonical import formatting")

    compiler_modules, compiler_errors = lean_direct_imports(path)
    errors.extend(compiler_errors)
    if compiler_modules != sorted(expected):
        errors.append(
            f"{label} crosses its approved dependency boundary: "
            f"Lean reports {compiler_modules}, expected {sorted(expected)}"
        )


def validate_axioms() -> list[str]:
    errors: list[str] = []
    completed = subprocess.run(
        ["lake", "env", "lean", "LeanDiracAudit.lean"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ["Lean axiom audit failed to execute:\n" + completed.stdout + completed.stderr]

    output = completed.stdout + completed.stderr
    seen: set[str] = set()
    for theorem in AUDITED_THEOREMS:
        escaped = re.escape(theorem)
        match = re.search(
            rf"'{escaped}' depends on axioms:\s*\[([^\]]*)\]", output
        )
        empty = re.search(rf"'{escaped}' does not depend on any axioms", output)
        if not match and not empty:
            errors.append(f"axiom audit did not report theorem: {theorem}")
            continue
        seen.add(theorem)
        dependencies = (
            {entry.strip() for entry in match.group(1).split(",") if entry.strip()}
            if match
            else set()
        )
        unexpected = dependencies - ALLOWED_AXIOMS
        if unexpected:
            errors.append(
                f"unapproved axiom dependency in {theorem}: {sorted(unexpected)}"
            )
    if seen != AUDITED_THEOREMS:
        errors.append("axiom audit theorem set is incomplete")
    return errors


def validate() -> list[str]:
    errors: list[str] = []
    lock = load(LOCK)
    status = load(STATUS)
    review = load(REVIEW)

    if lock.get("immutable") is not True or lock.get("canonical") is not False:
        errors.append("action lock must be immutable and noncanonical")
    if lock.get("field_families") != 2:
        errors.append("action does not declare exactly two field families")
    if status.get("highest_passed_gate") != "DIRAC-006":
        errors.append("status does not stop at DIRAC-006")
    if status.get("next_gate") != "DIRAC-007":
        errors.append("DIRAC-007 is not the next gate")
    if status.get("completed") != 6 or status.get("total") != 15:
        errors.append("gate completion counts are incorrect")
    if status.get("gates") != EXPECTED_GATES:
        errors.append("complete fifteen-gate status map is incorrect")
    if status.get("gate_modules") != EXPECTED_GATE_MODULES:
        errors.append("gate-to-module map is missing or incorrect")
    if review.get("status") != "PASS_ZERO_CRITICAL_HIGH":
        errors.append("no-bolt-on review is not cleared")
    if review.get("critical_findings") or review.get("high_findings"):
        errors.append("review contains critical/high findings")

    locked_files = lock.get("files", {})
    if set(locked_files) != EXPECTED_LOCKED_FILES:
        errors.append("action lock does not contain exactly the approved proof files")
    if lock.get("publication_theorem") != "LeanDirac.current_highest_gate_is_dirac006":
        errors.append("action lock does not name the DIRAC-006 publication theorem")

    for relative, expected in locked_files.items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"locked file missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"locked file changed: {relative}")

    production_paths = sorted((ROOT / "LeanDirac").rglob("*.lean"))
    production_paths.extend([ROOT / "LeanDirac.lean", ROOT / "LeanDiracAudit.lean"])
    actual_production_files = {
        path.relative_to(ROOT).as_posix() for path in production_paths
    }
    if actual_production_files != EXPECTED_LOCKED_FILES:
        errors.append(
            "production source tree differs from the exact approved file set: "
            f"extra={sorted(actual_production_files - EXPECTED_LOCKED_FILES)}, "
            f"missing={sorted(EXPECTED_LOCKED_FILES - actual_production_files)}"
        )
    production_text = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    )
    for forbidden in ("sorry", "admit", "axiom"):
        if re.search(rf"\b{forbidden}\b", production_text):
            errors.append(f"forbidden proof token present: {forbidden}")

    require_imports(
        ROOT / "LeanDirac.lean", ["LeanDirac.Publication"], errors, "public facade"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac001.lean",
        ["LeanDirac.C19Geometry"], errors, "DIRAC-001"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac002.lean",
        ["LeanDirac.Dirac001", "LeanDirac.PreparedBackground"], errors, "DIRAC-002"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac003.lean",
        ["LeanDirac.Dirac002"], errors, "DIRAC-003"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac003Contract.lean",
        ["LeanDirac.Dirac003"], errors, "DIRAC-003 definitional contract"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac004.lean",
        ["LeanDirac.Dirac003Contract"], errors, "DIRAC-004"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac004DiscreteAction.lean",
        ["LeanDirac.Dirac004"], errors, "DIRAC-004 discrete-action closure"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac005.lean",
        ["LeanDirac.Dirac004DiscreteAction"], errors, "DIRAC-005"
    )
    require_imports(
        ROOT / "LeanDirac" / "Dirac006.lean",
        ["LeanDirac.Dirac005"], errors, "DIRAC-006"
    )
    require_imports(
        ROOT / "LeanDirac" / "Publication.lean",
        ["LeanDirac.Dirac006"], errors, "publication root"
    )

    errors.extend(validate_axioms())

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("PRODUCTION_GUARD_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PRODUCTION_GUARD_PASS")
    print("highest_passed_gate=DIRAC-006")
    print("next_gate=DIRAC-007")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

