# DIRAC-029 red-team review

## Verdict

`PASS_SCOPED_FREE_CHARGE_CONJUGATION`

No critical or high finding remains. No native framework object changes.

## Injection and circularity checks

- The anti-linear map uses only complex conjugation, the inherited mass
  grading and the inherited y-transport generator.
- Lean proves the map squares to the identity and preserves density; neither
  property is a premise.
- The positive-to-negative-frequency relationship is proved against the same
  C19 Hamiltonian with reversed momentum.
- Both existing cubical flavor copies carry the map. No copy is deleted or
  relabeled as the antiparticle.
- No electric potential, charge sign, interaction rule or positron label is
  introduced.

## Adversarial controls

Removing conjugation destroys anti-linearity. Removing the mass grading or
using a real transport axis in place of the imaginary y generator breaks the
Hamiltonian intertwining. Failing to reverse momentum gives the wrong branch.

## Claim boundary

This proves the free Dirac charge-conjugation and negative-frequency partner.
It does not identify measured electric charge or prove interacting positron
phenomenology.
