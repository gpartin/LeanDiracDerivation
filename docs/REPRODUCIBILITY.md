# Reproducibility

## Formal verification

Install `elan`, clone the repository, and run:

```bash
lake update
python scripts/verify_release.py
```

`verify_release.py` performs the production proof build, compiled-environment
axiom audit, manifest checks, adversarial audit tests, and derivation-contract
audit. It is read-only and does not install hooks or change the Git index.

The exact Lean version is in `lean-toolchain`; Mathlib and transitive Lean
dependencies are pinned in `lake-manifest.json`.

## Numerical evidence

The numerical evidence is supplementary to the formal proof. Reproduce it in
an isolated Python environment:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt  # Windows
.venv/Scripts/python experiments/dirac009_native_coupling/dirac009_nondegenerate_stability.py
```

On POSIX systems use `.venv/bin/python`. The run is deterministic with seed
`120009`. It evolves a 4x4x4 periodic C19 lattice for 100 cycles at 64, 128,
and 256 updates per cycle and records pattern defects and relative energy
deviation after every update.

The evidence manifest records the script and output hashes, dependency
versions, platform, device class, and source provenance. Regenerating the JSON
requires an intentional review and manifest refresh.

## Updating manifests

After an intentional reviewed change only:

```bash
python scripts/update_manifests.py
python scripts/verify_release.py
```

The four manifests have different authority:

- `framework_lock.json`: native action-defining files only.
- `proof_manifest.json`: complete Lean proof/audit source set.
- `evidence_manifest.json`: numerical script, output, seed, and environment.
- `release_manifest.json`: documentation, tooling, workflow, and package inputs.

This separation prevents a documentation or audit edit from masquerading as a
change to the native action.
