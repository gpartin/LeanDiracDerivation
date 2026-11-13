"""Build and verify a deterministic publication archive from tracked files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "LeanDiracDerivation-DIRAC-016.zip"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


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

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    files = committed_files()
    payloads = {name: committed_payload(name) for name in files}
    manifest = "".join(
        f"{digest(payloads[name])}  {name}\n" for name in files
    ).encode("utf-8")

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as package:
        for name in files:
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            package.writestr(info, payloads[name])
        info = zipfile.ZipInfo("PACKAGE_MANIFEST.sha256", FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        package.writestr(info, manifest)

    with zipfile.ZipFile(archive, "r") as package:
        names = package.namelist()
        expected = files + ["PACKAGE_MANIFEST.sha256"]
        if names != expected:
            raise RuntimeError("publication archive file order or coverage differs")
        for name in files:
            if package.read(name) != payloads[name]:
                raise RuntimeError(f"publication archive changed {name}")
        if package.read("PACKAGE_MANIFEST.sha256") != manifest:
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
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
