# Related work and attribution

The geometric lattice-fermion ideas used in this repository are established
prior work. This project does **not** claim to have invented the Dirac
equation, Clifford algebra, the differential-form formulation of fermions,
Kähler--Dirac operators, lattice incidence calculus, fermion doubling,
Noether currents, Lorentz representation theory, the Pauli--Lubański vector,
or the Schrödinger equation.

The project-specific contribution is narrower: it gives a machine-checked
Lean chain from the declared two-field C19 action to the scoped results stated
in `docs/CLAIMS.md` and `docs/V32_RELEASE_NOTES.md`.

## Direct design source

The direct literature clue used to look for a Clifford algebra in oriented
cube data rather than insert standard Dirac matrices was Appendix A of:

- F. L. Teixeira, “Differential Forms in Lattice Field Theories: An
  Overview,” *International Scholarly Research Notices*, vol. 2013, Article
  ID 487270 (2013).
  [DOI 10.1155/2013/487270](https://doi.org/10.1155/2013/487270).

That appendix reviews the Dirac--Kähler description, Clifford products on
inhomogeneous differential forms, hypercubic-lattice transcription, and the
appearance of multiple lattice copies. Teixeira’s review is a design source,
not a coauthored result of this repository; the Lean definitions and proofs
here were written for this specific C19 action and checked independently.

## Dirac and differential-form fermion lineage

- P. A. M. Dirac, “The quantum theory of the electron,” *Proceedings of the
  Royal Society A* **117**(778), 610–624 (1928).
  [DOI 10.1098/rspa.1928.0023](https://doi.org/10.1098/rspa.1928.0023).
- D. Iwanenko and L. Landau, “Zur Theorie des magnetischen Elektrons. I,”
  *Zeitschrift für Physik* **48**, 340–348 (1928).
  [DOI 10.1007/BF01339119](https://doi.org/10.1007/BF01339119).
- E. Kähler, “Der innere Differentialkalkül,” *Rendiconti di Matematica e
  delle sue Applicazioni* **21**, 425–523 (1962).
- P. Becher and H. Joos, “The Dirac-Kähler equation and fermions on the
  lattice,” *Zeitschrift für Physik C* **15**, 343–365 (1982).
  [DOI 10.1007/BF01614426](https://doi.org/10.1007/BF01614426).

Ivanenko and Landau supplied the 1928 differential-form precursor; Kähler’s
later exterior-calculus formulation gave the construction its common modern
name. Becher and Joos developed the lattice Dirac--Kähler connection. The
repository’s `cubicalKahlerDirac` definition and incidence square belong to
this lineage.

## Lattice doubling and the exact scope here

- H. B. Nielsen and M. Ninomiya, “A no-go theorem for regularizing chiral
  fermions,” *Physics Letters B* **105**(2–3), 219–223 (1981).
  [DOI 10.1016/0370-2693(81)91026-1](https://doi.org/10.1016/0370-2693(81)91026-1).
- J. M. Rabin, “Homology theory of lattice fermion doubling,” *Nuclear
  Physics B* **201**(2), 315–332 (1982).
  [DOI 10.1016/0550-3213(82)90434-5](https://doi.org/10.1016/0550-3213(82)90434-5).

V3.2 proves a zero census for its exact scalar C19 symbol and classifies its
two cubical copies within the declared operator scope. It does not claim to
supersede the Nielsen--Ninomiya theorem, solve the general lattice-fermion
doubling problem, or construct an exact finite-spacing local four-component
Dirac square root. In fact, V3.2 proves that the current local continuum root
is not an exact global factorization of the finite-C19 symbol.

## Algebra, symmetry, and limiting structures

- W. K. Clifford, “Applications of Grassmann’s Extensive Algebra,” *American
  Journal of Mathematics* **1**(4), 350–358 (1878).
  [DOI 10.2307/2369379](https://doi.org/10.2307/2369379).
- E. Noether, “Invariante Variationsprobleme,” *Nachrichten von der
  Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische
  Klasse*, 235–257 (1918).
- M. Fierz and W. Pauli, “On relativistic wave equations for particles of
  arbitrary spin in an electromagnetic field,” *Proceedings of the Royal
  Society A* **173**, 211–232 (1939).
  [DOI 10.1098/rspa.1939.0140](https://doi.org/10.1098/rspa.1939.0140).
- E. P. Wigner, “On Unitary Representations of the Inhomogeneous Lorentz
  Group,” *Annals of Mathematics* **40**(1), 149–204 (1939).
  [DOI 10.2307/1968551](https://doi.org/10.2307/1968551).
- J. K. Lubański, “Sur la théorie des particules élémentaires de spin
  quelconque. I,” *Physica* **9**(3), 310–324 (1942).
  [DOI 10.1016/S0031-8914(42)90113-7](https://doi.org/10.1016/S0031-8914(42)90113-7).
- E. Schrödinger, “Quantisierung als Eigenwertproblem,” *Annalen der Physik*
  **384**(4), 361–376 (1926).
  [DOI 10.1002/andp.19263840404](https://doi.org/10.1002/andp.19263840404).

These sources provide the historical and mathematical setting for the named
Clifford, Noether, Lorentz/spin, Pauli--Lubański, irreducible-particle, and
Schrödinger-limit structures formalized downstream. Their inclusion does not
mean that the repository imports their conclusions as axioms: Lean checks the
stated identities from the repository’s explicit definitions and standard
foundations.

## Formal-verification infrastructure

The machine-checked artifact is implemented in Lean 4 and depends on mathlib.
Those projects supply the theorem prover, kernel, tactics, and general-purpose
formal mathematics used to check the repository; they are infrastructure, not
sources of the LFM action or the project-specific physical claims.

- L. de Moura and S. Ullrich, “The Lean 4 Theorem Prover and Programming
  Language,” in *Automated Deduction -- CADE 28*, 625–635 (2021).
  [DOI 10.1007/978-3-030-79876-5_37](https://doi.org/10.1007/978-3-030-79876-5_37).
- The mathlib Community, “The Lean Mathematical Library,” in *Proceedings of
  the 9th ACM SIGPLAN International Conference on Certified Programs and
  Proofs* (2020).
  [DOI 10.1145/3372885.3373824](https://doi.org/10.1145/3372885.3373824).

## What is specific to this repository

Starting from its declared reciprocal complex-`Psi`/complex-`chi` C19 action,
the repository formally derives its exact vacuum and tangent, identifies two
transverse C19 Klein--Gordon channels, constructs the oriented cubical
incidence root, proves the lossless four-component readout and two-copy
decomposition, and verifies the scoped free-Dirac, Lorentz/spin, spectral,
symmetry, nonrelativistic, and transport-reference results described in the
release documents.

That action-to-certificate integration, its exact Lean statements, and its
explicit finite-C19 obstruction/remainder analysis are the contribution being
claimed. The repository does not claim priority for the general mathematical
objects or physical equations cited above.
