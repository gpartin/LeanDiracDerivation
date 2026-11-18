# Related work and attribution

The geometric lattice-fermion idea used by this repository is established
prior work. This project does **not** claim to have invented the use of cell
data, differential forms, Clifford products, or staggered degrees of freedom
to represent Dirac fermions on a lattice.

## Direct design source

The direct source used during this project was Appendix A of:

F. L. Teixeira, “Differential Forms in Lattice Field Theories: An Overview,”
*International Scholarly Research Notices*, vol. 2013, Article ID 487270,
2013. [DOI 10.1155/2013/487270](https://doi.org/10.1155/2013/487270).

That appendix explains the Dirac--Kähler description, the Clifford product
on inhomogeneous differential forms, its relation to gamma-matrix
anticommutation, its hypercubic-lattice transcription, and the appearance of
multiple lattice copies. It was the design clue for looking for a Clifford
algebra in oriented cube data rather than inserting standard Dirac matrices.

## Foundational sources identified by that review

- P. Becher and H. Joos, “The Dirac-Kähler equation and fermions on the
  lattice,” *Zeitschrift für Physik C* **15**, 343–365 (1982).
  [DOI 10.1007/BF01614426](https://doi.org/10.1007/BF01614426).
- J. M. Rabin, “Homology theory of lattice fermion doubling,” *Nuclear
  Physics B* **201**(2), 315–332 (1982).
  [DOI 10.1016/0550-3213(82)90434-5](https://doi.org/10.1016/0550-3213(82)90434-5).

These sources precede this repository and must receive credit for the
geometric lattice-fermion lineage.

## Narrow contribution of this repository

The repository starts from its declared two-field C19 action and proves a
specific conditional chain for its retained eight-corner sector. DIRAC-017
specializes the established Clifford-geometric idea to the signed incidence
operators already derived earlier in that chain. Lean then verifies that:

- their oriented plane products support three self-adjoint rotation
  generators;
- the standard rotation commutator has a unique nonzero real normalization,
  one half;
- the component square and Casimir are one quarter and three quarters;
- both retained copies intertwine with the conventional four-component spin
  generators and have the explicit two-plus/two-minus component spectrum;
- the native inner product is positive definite and the continuous rotations
  are unitary;
- covariance and unitarity uniquely characterize the positive quarter-turn
  inside the native scalar-plus-plane rotor family;
- a `2π` rotation gives a sign reversal and a `4π` rotation returns the
  state; and
- the same rotations commute with the mass grading and act covariantly on the
  full interpreted mass-plus-transport Hamiltonian.

Those exact definitions, proofs, and their integration into the transitive
LFM action-to-Dirac certificate are the project-specific formal contribution.
They do not establish priority for Clifford spinors, Dirac--Kähler fermions,
staggered fermions, fermion doubling, or spinorial double-valued rotations.
They also do not prove that unrestricted native time evolution dynamically
applies this representation to a localized stable LFM electron.
