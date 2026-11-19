# SRIM stopping tables

Read with `isotopia_py.srim` (`load_table`). The files are SRIM output, used under SRIM's
non-commercial terms and not covered by the MIT licence; see `NOTICE`.

| Folder | Projectile (ISOTOPIA symbol) | Files |
|---|---|---|
| `p/` | proton (`p`) | `p_in_<Symbol>.txt`, e.g. `p/p_in_Ni.txt` |
| `d/` | deuteron (`d`) | `d_in_<Symbol>.txt` |
| `t/` | triton (`t`) | `t_in_<Symbol>.txt` |
| `h/` | helium-3 (`h`) | `h_in_<Symbol>.txt` |
| `a/` | alpha (`a`) | `a_in_<Symbol>.txt`, e.g. `a/a_in_Ni.txt` |

`index.csv` lists every element with the density, target mass and phase used.

## Elements

All 92 elements from H to U, the targets SRIM accepts (it takes no heavier element):

- The 84 with a CIAAW standard atomic weight: Z = 1–83 except Tc and Pm, plus Th, Pa and U
  (CRC 107th ed., Section 1, *Standard Atomic Weights*, footnote u).
- **Tc, Pm, Po, At, Rn, Fr, Ra, Ac** have no stable isotope. Their target mass is the
  longest-lived isotope in ISOTOPIA's decay library: Tc-98, Pm-145, Po-209, At-210, Rn-222,
  Fr-223, Ra-226, Ac-227. Densities are the CRC's at 25 °C except At and Fr (below); the CRC
  gives Tc to 2 and Ra (5) and Ac (10) to 1 significant figure. Rn is run as a gas.
- **Neighbour check:** above 10 MeV, the electronic stopping per atom divided by Z of each of
  these lies within the run of its neighbours (Tc between Mo and Ru, Pm between Nd and Sm,
  Po to Ac between Bi and Th). Rn is the exception: as a gas, SRIM gives it about 5% more
  alpha stopping than its solid neighbours at 10 MeV, the same gas-phase effect it gives Xe
  (about 3%).
- **At and Fr.** Their densities are estimates at "room temperature" (25 °C is not stated),
  Arblaster, *Selected Values of the Crystallographic Properties of the Elements*, ASM
  International (2018): At 8.95 g/cm³, Fr 2.458 g/cm³. Their target masses are those of the
  longest-lived isotopes in ISOTOPIA's decay library, At-210 (209.987 u) and Fr-223
  (223.020 u). No bulk sample of either element exists, so SRIM has no measured stopping
  data for them: its values come from its own interpolation between neighbouring elements.
  Above 10 MeV they follow the neighbours smoothly (electronic stopping per atom divided by
  Z changes by under 2% from Bi to Th). Below about 1 MeV the per-atom values differ by up to
  about 20% between neighbouring elements, Pb and Bi included, so treat the low-energy end
  as rough.

## SRIM settings

| SR.IN field | Value |
|---|---|
| Ion | ISOTOPIA's `parZ` and `parmass` (`constants.f90` lines 46 and 48): p = Z 1, 1.00782495281 u; d = Z 1, 2.01410195518 u; t = Z 1, 3.01604916211 u; h = Z 2, 3.01602879487 u; a = Z 2, 4.00260293529 u. SRIM labels them "Hydrogen" and "Helium" in the header. |
| Target mass | `mass_u` in the group's `element_data.csv`: CIAAW standard atomic weight (abridged value where CIAAW gives an interval) |
| Density | `density_g_cm3` in `element_data.csv`: 25 °C, gases at 101.325 kPa. Equal to `constants.RHOTABLE` (checked in `tests/test_srim.py`). |
| Phase | SRIM gas flag for H, He, N, O, F, Ne, Cl, Ar, Kr, Xe, Rn; solid for all others |
| Compound correction | 1 (none; single elements) |
| Stopping units | eV/(1E15 atoms/cm2) (SR Module code 7) |
| Energies | 1 keV to 250 MeV (ISOTOPIA's beam limit, `checkvalue.f90` line 102), on SRIM's own grid: 142 energies, the same in every file. The first prints as "999.999 eV". |

Allotropes use the data-sheet defaults: graphite C, red P, alpha S, gray Se, white Sn
(gray As, gray Sb).

**Br and Hg are liquids at 25 °C. Their tables are SRIM *solid* calculations using the liquid
density.**

Only the two dE/dx columns (electronic, nuclear) are used. They are per atom and do not depend
on density. Range and straggling depend on SRIM's density and are not used.

## How they were made

- **Program:** SRIM's batch stopping program, `SR Module/SRModule.exe`, from a SRIM-2013
  installation, run under Wine.
- **Header label:** the headers say "SRIM-2012.01 / Calculation using SRIM-2006", because
  that is the SR Module's own version label.
- **Same output as SRIM-2013:** with the same inputs, it reproduced three SRIM-2013.00 outputs
  from the stopping-table window digit for digit. These were H in Ni at default and at changed
  mass and density, and Fe in Fe at 3.5 MeV.
- **Rebuilding:**
  `python scripts/srim/make_tables.py p element_data.csv "<SRIM>/SR Module" <out>` rebuilds
  the proton set (`a` for alphas) and writes each exact `SR.IN` next to the tables.
  `scripts/srim/check_tables.py` repeats the checks below.

## Checks

- Every header's density and ion mass match the inputs, and the target mass implied by the
  header matches the sheet.
- Our conversion to MeV/cm matches the MeV/mm factor SRIM prints in each file, to its 5 digits
  (`tests/test_srim.py`).
- Electronic stopping vs ISOTOPIA's Bethe-Bloch (`stoppingpower.f90`) at 50–250 MeV, Z ≥ 13:
  - p: 0.967–1.026
  - d: 0.949–1.027
  - t: 0.936–1.025
  - h: 0.943–1.031
  - a: 0.934–1.029

  For Z < 13, SRIM is higher than ISOTOPIA's Bethe-Bloch by up to 17% (protons in H),
  because `stoppingpower.f90` uses its Z ≥ 13 mean-excitation-energy formula for every
  element.
- **Velocity scaling:** ions of the same charge at the same velocity have the same electronic
  stopping. The d and t tables at energy E × m/m_p agree with the p table at E, and He-3 with
  the alpha table the same way, to 0.38% or better from 10 keV to 80 MeV in all 92 elements
  (`tests/test_srim.py`). Nuclear stopping depends on the ion mass and does not scale this way.
- **Interpolation:** `isotopia_py.srim` interpolates linearly in log E and log S. As a test, I
  dropped each table row in turn and predicted it from its neighbours, which doubles the
  spacing:
  - median of the per-table worst errors: 0.5%
  - every table under 1.6%, except p in He near 2 keV (3.6%)
  - the error at the real spacing is smaller.
