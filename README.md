# ISOTOPIA Python starter

This is a first, intentionally limited Python port based on the provided
ISOTOPIA 2.23 Fortran source and the supplied `isotopia.inp`, `isotopia.out`,
and `At211.act` example.

## Included

- `isotopia_py/io.py`
  - reads basic keyword/value input files (`project` and `projectile` aliases)
  - reads the seven numeric columns in isotope-specific `.act` files
  - preserves header lines and extracts metadata values where possible
- `isotopia_py/kinetics.py`
  - translates the direct target depletion and direct product-production
    equations in `source/equations.f90`
  - translates `activity = decay_constant * inventory` from
    `source/activities.f90`
- `inspect_act.py`
  - command-line example for inspecting an input and `.act` output file

## Try it

From this directory:

```bash
python inspect_act.py --input "/path/to/isotopia.inp" --act "/path/to/At211.act"
```

The package uses only the Python standard library.

## Important scope limits

This is **not a full replacement for ISOTOPIA**. In particular, the
simplified kinetics functions do not include:
- feeding from radioactive parent nuclides or complete decay chains
- isomeric transitions and branch ratios
- energy-dependent cross-section interpolation and spectrum averaging
- stopping-power / energy-loss integration through the target
- the Fortran code's full default handling, output scaling, or data-library logic

The next modules to port for a meaningful end-to-end calculation are:
1. input defaults and validation (`input.f90`, `checkvalue.f90`)
2. cross-section file reading/interpolation (`crosssections.f90`, `pol1.f90`, `locate.f90`)
3. spectrum-averaged reaction rates (`reactionrates.f90`)
4. decay data and parent feeding (`decaydata.f90`, the parent-decay terms in `equations.f90`)
5. output formatting (`prodout.f90`)

Validation should compare Python outputs against ISOTOPIA's `.act` time series for multiple samples, not just one nuclide.
