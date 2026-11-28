# V3.6 Hostile Red Team 1: Technical Derivation

Date: 2026-11-28

## Scope

This review attacked the revised action, its geometric normalization, the
connection to the inherited free-Dirac theorem, and the nonlinear-response
claims. The release criterion was zero unresolved critical or high findings.

## Findings and repairs

### High: the coefficient was initially a literal

The first implementation set `chiKineticNormalization := 1` and proved the
Hodge isometry afterward. A reviewer could fairly describe that as inserting
the desired number and supplying a retrospective justification.

Repair: `Foundation.lean` now defines the coefficient as the actual norm ratio
of a corresponding cubical vector/bivector unit-blade pair. Lean evaluates
that expression to one. Stage 1 separately proves the full Hodge isometry and
that every coordinate unit blade has the same live relative weight.

Status: RESOLVED.

### High: cube geometry could be overclaimed as universally forcing `B = 1`

The Hodge isometry alone does not prove that every conceivable two-field
action must assign one common kinetic metric to both representations.

Repair: the action-design premise is explicit: V3.6 selects one standard
Euclidean cubical Clifford kinetic metric for the two grades. Lean proves
`B = 1` uniquely *within that metric class*. The arbitrary-`B` common-turn
theorem prevents rotation symmetry alone from being reported as a selector.

Status: RESOLVED by scoped claim, not by a universal uniqueness theorem.

### Medium: the radial model duplicated a polynomial without an action link

The first radial-response stage introduced a scalar potential with the same
formula as the native local potential but did not prove that it was a slice of
the action.

Repair: Stage 5 now constructs an explicit one-site native state and proves
that its `sitePotential` is exactly the scalar radial potential before deriving
the response consequences.

Status: RESOLVED.

### Medium: field ontology and grade representation were conflated

Some prose said that `Psi` and `chi` *are* vector and bivector fields, while
the native state still stores two `C^3` arrays and the action's `wedge` accepts
two such arrays.

Repair: code and documentation now state that the native fields remain
distinct `C^3` arrays. The Hodge map represents `chi`'s coordinates in the
bivector grade for the metric comparison; it does not identify or merge the
fields.

Status: RESOLVED.

## Final technical verdict

Critical findings: 0 unresolved.

High findings: 0 unresolved.

The formal result is defensible as a geometrically motivated revised research
action with scoped uniqueness and a complete free-Dirac no-regression proof.
It is not a proof that all possible kinetic architectures select `B = 1`, nor
is it nonlinear electron or gravity validation.
