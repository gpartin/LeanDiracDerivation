"""Build and verify a deterministic publication archive from tracked files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "LeanDiracDerivation-DIRAC-025.zip"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
PROVENANCE_NAME = "PROVENANCE.json"
MANIFEST_NAME = "PACKAGE_MANIFEST.sha256"


def require_clean_tracked_tree() -> None:
    """Reject an archive that would silently omit tracked working-tree edits."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "tracked working tree is not clean; commit or restore changes "
            "before building the publication package"
        )


def committed_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=ROOT,
        text=True,
        capture_output=True, check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def committed_payload(name: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{name}"], cwd=ROOT,
        capture_output=True, check=True,
    ).stdout


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / ARCHIVE_NAME

    require_clean_tracked_tree()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    files = committed_files()
    payloads = {name: committed_payload(name) for name in files}
    framework = json.loads(payloads["project/framework_lock.json"])
    status = json.loads(payloads["project/status.json"])
    proof = json.loads(payloads["project/proof_manifest.json"])
    provenance = (json.dumps({
        "schema_version": "1.0.0",
        "archive": ARCHIVE_NAME,
        "git_commit": commit,
        "lean_toolchain": payloads["lean-toolchain"].decode("utf-8").strip(),
        "action_id": framework["action_id"],
        "highest_gate": status["highest_passed_gate"],
        "highest_gate_status": status["gates"][status["highest_passed_gate"]],
        "publication_theorem": proof["publication_theorem"],
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest = "".join(
        f"{digest(payloads[name])}  {name}\n" for name in files
    )
    manifest += f"{digest(provenance)}  {PROVENANCE_NAME}\n"
    manifest = manifest.encode("utf-8")

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as package:
        for name in files:
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            package.writestr(info, payloads[name])
        info = zipfile.ZipInfo(PROVENANCE_NAME, FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        package.writestr(info, provenance)
        info = zipfile.ZipInfo(MANIFEST_NAME, FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        package.writestr(info, manifest)

    with zipfile.ZipFile(archive, "r") as package:
        names = package.namelist()
        expected = files + [PROVENANCE_NAME, MANIFEST_NAME]
        if names != expected:
            raise RuntimeError("publication archive file order or coverage differs")
        for name in files:
            if package.read(name) != payloads[name]:
                raise RuntimeError(f"publication archive changed {name}")
        if package.read(PROVENANCE_NAME) != provenance:
            raise RuntimeError("publication archive provenance differs")
        if package.read(MANIFEST_NAME) != manifest:
            raise RuntimeError("publication archive manifest differs")

    archive_hash = digest(archive.read_bytes())
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "status": "PUBLICATION_PACKAGE_PASS",
        "archive": str(archive),
        "sha256": archive_hash,
        "commit": commit,
        "committed_files": len(files),
        "provenance": PROVENANCE_NAME,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
