DIRAC-026 PASS: the derived electron sector now knows how to boost

We have extended the Lean chain from rotations to motion between inertial
frames. The three boost operations were not inserted as new rules: Lean built
them from the three direction-of-motion operators already derived from the
C19 lattice.

It then proved that rotations and boosts obey the Lorentz algebra, that every
boost has the correct inverse, and that boosting either rest-spin state gives
a moving state of the same Dirac Hamiltonian. The transformed energy,
momentum, density and current all agree, while

    energy^2 - c_eff^2 momentum^2 = chi0^2

stays unchanged. Here `chi0 = 19` is the rest coefficient inherited from the
framework; nothing was fitted to force the identity.

This is a continuum-sector result, not a claim that a finite cubic grid has
perfect Lorentz symmetry at every wavelength. But it closes an important
electron-only gap: the same operator that supplied spin and transport also
supplies the correct relativistic change of frame.

Next, DIRAC-027 asks whether these boosts and rotations combine into the
frame-independent spin-one-half invariant used to classify a relativistic
electron.
