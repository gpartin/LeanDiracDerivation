# Contributing

Contributions are welcome as issues, discussions, or focused pull requests.

Before opening a pull request:

1. state the exact theorem or documentation boundary being changed;
2. distinguish free linearized mathematics from nonlinear or physical-electron
   claims;
3. do not introduce measured electron inputs or unproved physical closure;
4. preserve historical release and external-submission artifacts; and
5. run `python scripts/verify_release.py` from the repository root.

New production theorems must compile without `sorry`, custom axioms, or opaque
proof substitutes. Documentation must identify assumptions, controls,
remainders, and non-claims explicitly. Independent physics and Lean review is
especially valuable.
