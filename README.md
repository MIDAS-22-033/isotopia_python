# ISOTOPIA in Python

A line-for-line Python port of [ISOTOPIA 2.23](https://github.com/arjankoning1/isotopia),
A.J. Koning's code for predicting radio-isotope production yields. It reads the
same input files and the same data library (`isotopia.libs`) and writes the
same output files, so the two codes can be compared directly. On the 18 sample
cases shipped with ISOTOPIA the output is identical to the Fortran, digit for
digit, apart from one file where the Fortran itself differs between platforms
(see *Precision* below).

The point of the port is to make the physics easy to change. Each Fortran
source file maps onto one Python function or module, and the data flows through
one `State` object instead of global arrays.

## Layout

| Module | Fortran source | What it does |
|---|---|---|
| `isotopia_py/run.py` | `isotopia.f90`, `isotopiainitial.f90`, `reacinitial.f90`, `production.f90`, `natural.f90` | main program and the loop over isotopes of a natural element |
| `isotopia_py/inputdeck.py` | `machine.f90`, `readinput.f90`, `convert.f90`, `getkeywords.f90`, `input.f90`, `checkkeyword.f90`, `checkvalue.f90`, `checkfiles.f90`, `conversion.f90`, `inputout.f90` | keywords, defaults, checks, unit factors, input echo |
| `isotopia_py/data.py` | `abundance.f90`, `decaydata.f90`, `crosssections.f90` | abundances, half-lives and decay modes (ENDF files), cross-section tables |
| `isotopia_py/rates.py` | `stoppingpower.f90`, `reactionrates.f90` | beam spectrum, Bethe-Bloch stopping power, self-shielding, spectrum-averaged reaction rates |
| `isotopia_py/kinetics.py` | `equations.f90`, `activities.f90` | target depletion, production, decay and parent feeding on the time grid; activities and yields |
| `isotopia_py/output.py` | `prodout.f90`, `natural.f90` (output part) | summary table, `.act` files, `elements.out` |
| `isotopia_py/yandf.py` | `write_block.f90` | YANDF `#` header blocks |
| `isotopia_py/constants.py` | `constants.f90`, `reacinitial.f90`, `A0_isotopia_mod.f90` | particles, element symbols, default densities (`rhotable`), time units, array sizes |
| `isotopia_py/state.py` | `A0_isotopia_mod.f90` | the `State` dataclass holding every run variable |
| `isotopia_py/fortran.py` | – | gfortran number formatting, list-directed reads, `locate`/`pol1`, float32 helpers |
| `isotopia_py/compare.py` | `samples/verify` | numeric comparison of two output directories |
| `isotopia_py/io.py` | – | readers for `.inp` and `.act` files, for post-processing |
| `isotopia_py/decaylib.py` | – (new) | every decay mode, branching ratio and daughter isomeric state from the same decay files |
| `isotopia_py/matrix.py` | – (new) | general Bateman solver N(t) = exp(Mt) N(0): decay network plus reaction terms |
| `isotopia_py/irradiation.py` | – (new) | irradiation + cooling of a target with the matrix solver, rates taken from a port run |
| `isotopia_py/daughters.py` | `reactionrates.f90` (weights) | reactions on products (multiple hits) inside an A/Z window, rates from the target's energy grid and weights |
| `isotopia_py/isotopia_compat.py` | `equations.f90` | ISOTOPIA-compatible mode: the port's equations, defects included, as one matrix exponential (for validation) |
| `isotopia_py/srim.py`, `isotopia_py/srim_tables/` | – (new) | SRIM stopping tables for `p`, `d`, `t`, `h` and `a` in the 92 elements from H to U, converted to MeV/cm; the default stopping power (see items 7 and 9 below) |
| `isotopia_py/energyloss.py` | – (new) | energy loss through a layer: thickness from (E_in, E_out) and exit energy from (E_in, thickness); used for the `thickness` input (see items 8 and 9 below) |

## Setup

Python 3.10+ with NumPy (and SciPy for `matrix.py`). The port needs ISOTOPIA's `files/` folder
(abundances, decay data, neutron and photon spectra) and the cross-section library. It stands
alone: `files/` and `samples/` are part of this repository, and only the library has to be
downloaded into it:

```text
isotopia_python/        # this repository
├── isotopia_py/        # the code, with the SRIM tables in isotopia_py/srim_tables/
├── files/              # ISOTOPIA data: abundance/, decay/, neutron_spectra/, photon_spectra/
├── samples/            # the 21 ISOTOPIA sample cases with Fortran reference outputs (org/)
└── isotopia.libs/      # github.com/arjankoning1/isotopia.libs (about 5 GB, 800 000 files)
```

`files/` and `samples/` are copied unchanged from the team's Fortran repository,
[MIDAS-22-033/isotopia_midas](https://github.com/MIDAS-22-033/isotopia_midas) (commit
`640f478`), which carries the original ISOTOPIA data and samples by A. J. Koning under the
MIT licence. The sample references (`samples/*/org`) are its fixed Fortran outputs; never
edit them.

Get the library once, from inside this repository (it is in `.gitignore` and is never
committed):

```bash
git clone --depth 1 https://github.com/arjankoning1/isotopia.libs
```

The code looks for the data in this order (`inputdeck.data_directories`):

1. the environment variables `ISOTOPIA_DIR` (the folder that holds `files/`) and
   `ISOTOPIA_LIBS` (the library), exactly as for the Fortran;
2. `files/` and `isotopia.libs/` inside this repository;
3. the Fortran's layout: a folder `isotopia` next to this repository, and `isotopia.libs`
   next to that.

`ISOTOPIA_USER="Your Name"` is optional and is written in output headers.

## Running

Same conventions as the Fortran: input on standard input (or as an argument),
summary on standard output, result files in the current directory (or `-o`):

```bash
python -m isotopia_py isotopia.inp -o results > results/isotopia.out
```

From Python:

```python
from isotopia_py import run
st = run(["projectile p", "element Mo", "mass 100", "Ebeam 16.", "Eback 12.", "Ibeam 0.15"],
         outdir="results")
st.activity[(43, 99, 1)]        # Tc-99m activity [Bq] on the 100-point time grid
st.Tgrid                        # the time grid [s]
st.reaction_rate[(43, 99, 1)]   # reaction constant [s^-1]
```

Nuclides are keyed `(Z, A, isomer)` with isomer `-1` = total, `0` = ground state,
`1` = first isomer, `2` = second.

## Matrix solver

`matrix.py` is the general solver for multiple hits. It stacks all nuclides in
a vector and solves dN/dt = M N exactly with a matrix exponential (manual
Eq. 4.1), so the time grid only chooses output points. M holds every decay
branch (isomeric transitions, chains of any length, read by `decaylib.py`)
and, during irradiation, the reactions on the target and, optionally, on its
products inside an A/Z window. The main ISOTOPIA run does not use it.

Decay only:

```python
from isotopia_py.matrix import decay
inv = decay({(42, 99, 0): 1e15}, times=[0, 3600, 86400])   # Mo-99, times in s
inv[(43, 99, 1)]            # Tc-99m atoms at each time
inv.activity((43, 99, 1))   # Tc-99m activity [Bq]
inv.warnings                # truncated chains, untracked fission, data issues
```

Irradiation, with the beam, target and reaction rates of a port run:

```python
from isotopia_py import run
from isotopia_py.irradiation import Irradiation, irradiate, isotopia_times
st = run(open("isotopia.inp").read().splitlines(), outdir="results")
irr = Irradiation.from_state(st)                 # target, N0, R_T, R_T->i, times
inv = irradiate(irr, isotopia_times(st))         # ISOTOPIA's 101 output times
inv[(43, 99, 1)]                                 # Tc-99m atoms
inv.warnings                                     # also rate inconsistencies
```

Differences from ISOTOPIA in this mode are listed in `irradiation.py`: full
decay feeding (defects 1-4), target decay and refeeding of the target by
decay, and state-resolved isomer rates. A product missing from the decay
library stops the run with the file and nuclide; pass `on_missing="warn"` to
keep it as a chain end with a warning. For a natural target, `run` keeps only
the last isotope, so `from_state` has to be called per isotope.

Multiple hits: every nuclide inside an A/Z window around the target that has
cross sections in the library is hit too. Its rates are ISOTOPIA's own sum
over the energy grid and weights (rebuilt from the port and checked against
every target rate), applied to its library files:

```python
from isotopia_py.daughters import Window, irradiate_with_hits
inv = irradiate_with_hits(st, Window(A_below=8, A_above=2, Z_below=3, Z_above=2))
inv[(43, 98, 0)]               # Tc-98, made only by second hits at 12-16 MeV
inv[(0, 0, 0)]                 # atoms that left the window ("outside_window")
```

Hit nuclides deplete the way ISOTOPIA actually depletes the target: at the
non-elastic rate for charged particles and photons, and at the total cross
section (elastic included) for neutrons with self-shielding. The library has unstable nuclides as targets only if they live about
a day or longer; shorter-lived ones are not hit, and a warning lists them.
Assumptions and limits are in the `daughters.py` docstring.

`isotopia_compat.compat_inventory(st)` reproduces the port's own equations,
defects included, with the same matrix machinery. It exists to check the
solver against the port and is not a physics model.

Tests: `tests/test_matrix.py` (decay), `tests/test_irradiation.py`
(closed-form irradiation cases; compatibility mode against the port on every
sample; full mode against the port for products that neither code feeds by
decay) and `tests/test_daughters.py` (multiple hits: closed forms, the rebuilt
weights against every port rate, perturbation theory on p + Mo-100). The
Mo-100 hit tests need the daughters' library folders (`p/Mo099`, `p/Tc099`,
...). Run them with `python -m pytest`.

## Checking against the Fortran

```bash
python scripts/run_samples.py                 # all samples vs samples/*/org
python scripts/run_samples.py p-Mo100-Tc099m  # one sample
python scripts/run_samples.py --ref runs/fortran   # vs your own Fortran runs
pytest                                        # same, plus unit tests
```

`scripts/run_samples.py` writes the Python results to `runs/python/<case>/` and
reports, per file, whether it is identical to the reference or within tolerance.

The port's defaults differ from the Fortran's in two places: the density table (item 3
below) and the stopping power (item 9). The sample checks therefore run with the Fortran's
settings, `run(..., fortran_defaults=True)`. From the command line that is
`python -m isotopia_py isotopia.inp --fortran-defaults`.

## Precision

ISOTOPIA stores nearly everything in single precision (`real(sgl)`) and solves
the production equations in double. The port keeps the same precision in the
same places (`numpy.float32` for `sgl` quantities), which is what makes the
output identical rather than merely close. Three consequences for anyone editing
the code:

* Keep `S(...)` (float32) on quantities that are `sgl` in the Fortran if you
  want to stay digit-identical. If you don't care about that, you can drop them.
* NumPy 1.x returns float64 for `float32 ** 2`; write `x * x` instead.
* Two correct builds still differ in the last digit where a result is the small
  difference of large numbers. `Ru099.act` in the Mo-100 sample is such a case:
  a stable nuclide fed only by Tc-99 decay (half-life 2e5 years) — the macOS
  and Linux Fortran builds disagree there too. `compare.py` allows for this
  kind of noise (see its docstring).

## Differences from the Fortran

The first commit of the port (`d7794d2`) reproduces the Fortran exactly,
including two bugs. Each is fixed in its own later commit so the effect can be
seen in the history:

1. **`xsfile` isomer index** (`input.f90`): the file name was stored under the
   isomer index read from the *previous* `xsfile` line, because the index is
   read after it is used. No sample uses `xsfile`, so outputs are unchanged.
2. **Stale inventories for natural targets** (`equations.f90`,
   `activities.f90`): the per-nuclide arrays are only zeroed for nuclides the
   current isotope run produces. A nuclide made by one isotope but not the
   next kept its old inventory, which the next run (a) added to the natural
   totals again, weighted by *its* abundance, and (b) used as a decaying parent
   during cooling. For natural Ga, nuclides made only by Ga-69 came out
   1/0.601 = 1.66 times too high, and some cooling-phase inventories were off
   by orders of magnitude (Cu-66). The port clears the inventories at the start
   of every isotope run; `tests/test_samples.py::test_natural_abundance_weighting`
   checks that the merged result is the abundance-weighted sum of the
   per-isotope results. Single-isotope targets are unaffected. The Fortran in
   `isotopia_midas` has the same fix (`equations.f90` clears `Niso` and
   `Nisorel` at the start of each run), and the references for the two
   natural-target samples, `p-Ga000-Ge068` and `p-Rb000-Sr082`, were
   regenerated with it, so the port matches them like every other sample.

Later changes that are not in the Fortran:

3. **Default densities** (`reacinitial.f90`, `rhotable`): the Fortran table
   gives no source or temperature, and mixes conventions (gases at 0 °C;
   graphite at its ideal-crystal density, 2.267 g/cm^3; neon at `9.0e-3`,
   ten times the density of neon gas). `constants.RHOTABLE` now holds the
   group's element data table, for Z = 1–99: values at 25 °C (gases at
   101.325 kPa) from the CRC Handbook (107th ed.) density table or the
   ideal-gas law (Br: from Marcus 2016's density equation at 25 °C, since the
   CRC value has no temperature; Os and Ir: from fits to Arblaster's density
   tables at 25 °C, since the CRC values are at 20 °C; Am: from the 1962
   lattice-parameter equations of McWhan et al. at 25 °C, since the CRC density
   table's 12 is not supported by the lattice; Bk: X-ray densities for Bk-247
   from the lattice parameters at 298.15 K in Morss et al. 2006, Table 15.8,
   because the CRC density table lists only beta-Bk; Es: X-ray density for
   Es-252 from the same table, since the CRC gives none). Not confirmed at 25 °C,
   as listed in `constants.py`: At and Fr (Arblaster 2018, room temperature, both
   estimates). Every sample takes its
   density from this table, so `tests/test_samples.py` runs the samples with
   `fortran_defaults=True`, which uses the original table
   (`constants.RHOTABLE_FORTRAN`), to keep checking the port against the
   Fortran reference outputs.
4. **No unverified densities**: Fm to Og (Z = 100–118) have no measured
   density, so they have no default; the input must give `rho`,
   otherwise the run stops with an error. (The Fortran used a placeholder of
   13.5.) No sample target is affected.
5. **Elements up to Z = 118**: the Fortran tables run to Z = 124, with the
   labels `B9`, `C0`–`C4` for the unmade elements 119–124. These are removed
   (`NUMZ = 118`), and `element` with Z > 118 is an input error.
6. **Allotrope message**: when the target density is taken from the table
   and the element has more than one solid form in the group's element data
   table (C, P, S, Se, Sn, Bk), the summary output lists the other densities and
   says which default is used, so the user can choose another with `rho`.
   No sample target is one of these elements.
7. **SRIM stopping tables** (new):
   - `isotopia_py/srim_tables/` holds SRIM stopping per atom, in eV/(1E15 atoms/cm2), for
     protons, deuterons, tritons, helium-3 and alphas in the 84 elements with a standard
     atomic weight and the 8 radioactive elements up to U (all 92 elements from H to U),
     from 1 keV to 250 MeV.
   - `isotopia_py.srim` reads them and converts to MeV/cm for a given density and atomic mass:
     `load_table("p", 42).stopping_power(E, rho, mass)`.
   - They are meant to replace the Bethe-Bloch formula (`stoppingpower.f90`), which fails at
     low energy. Item 9 says when they are used.
   - The tables are SRIM output, under SRIM's non-commercial terms and not covered by the MIT
     licence; see `isotopia_py/srim_tables/NOTICE`. `isotopia_py/srim_tables/README.md` gives
     how they were made and checked; `scripts/srim/` rebuilds them.
8. **Energy loss through a layer** (new):
   - The Fortran takes `Ebeam` and `Eback` and returns the thickness, L = ∫ dE/S (manual
     Eq. 4.23). `isotopia_py.energyloss.RangeTable` solves the same relation both ways:
     `thickness(E_in, E_out)` and `exit_energy(E_in, thickness)`. They are exact inverses.
   - Usage: `RangeTable.from_srim(srim.load_table("p", 42), rho, mass).exit_energy(16.0, 0.02)`.
   - Checks (`tests/test_energyloss.py`): the round trip holds for all 168 SRIM tables; with
     ISOTOPIA's own Bethe-Bloch stopping power it reproduces the thickness in the
     `p-Mo100-Tc099m` reference output (2.097183E-02 cm) and returns `Eback` = 12 MeV from it.
   - Model: mean energy loss only (no straggling), thickness measured along the track, one
     uniform material per layer. A particle that cannot cross the layer gets exit energy 0.
     The module docstring gives the assumptions and their size.
9. **`thickness` as an input and the `stopping` keyword** (charged particles; new):
   - **`thickness`**: a charged-particle target can be given as `Ebeam` and `thickness` [cm]
     in place of `Ebeam` and `Eback`. `Eback` is then calculated (item 8) and the run
     proceeds as before. Giving both `Eback` and `thickness` is an error. For neutrons and
     photons `thickness` means what it always did.
   - **`stopping srim` or `stopping bethe`**: the stopping-power model. The default is the
     SRIM tables where one exists (`p`, `d`, `t`, `h` and `a` in the 92 elements from H to U). Elsewhere the Bethe-Bloch
     formula is used and a warning is written. `fortran_defaults` makes Bethe-Bloch the default.
   - **The two inputs agree**: a run with `Ebeam` and `Eback` prints an effective target
     thickness; a run with `Ebeam` and that thickness returns the same `Eback` and the same
     results (`tests/test_thickness.py`, and `tests/test_samples.py` with the library).
   - **Effect of SRIM**: for `p-Mo100-Tc099m` the target for 16 → 12 MeV is 0.02159 cm
     (Bethe-Bloch with the same density: 0.02101 cm), and the Tc-99m activity is 2.7% higher.
   - **Beam stopping in the target** (thickness beyond the range): with SRIM, `Eback` is set to
     the lowest table energy (1 keV), a warning gives the depth reached, and only that depth
     counts as the effective target. With Bethe-Bloch the run stops with an error, because
     the formula is not valid at low energy.
   - **Atoms per cm3** are N_A ρ / M. M is the mass number of the target isotope, as in the
     Fortran. For a natural target (`mass 0`) it is the abundance-weighted mass number, so
     all isotopes of the element share one material, one `Eback` and one thickness. The
     Fortran's own configuration keeps each isotope's mass number, to reproduce its outputs.
     For an enriched target, give its real density with `rho`.
   - **Changed modules**: `inputdeck.py` (keywords, checks, input echo), `rates.py`
     (`setup_stopping`, beam spectrum), `run.py`, `state.py`, `kinetics.py` (atoms per cm3),
     `output.py` (two summary lines), `__main__.py`, `constants.py`.
10. **Numbered layers around the target** (`layer`; charged particles; new):
   - **Input**: `layer <n> <element> <thickness in cm> [density in g/cm^3]` for each capsule
     layer, and `layer <n> target` to say which layer is the target. The beam crosses layer
     1 first, then 2, 3, ...; the numbers must run 1, 2, 3, ... with each used once, and the
     lines can be in any order. The target's material still comes from `element`, `mass` and
     `rho`, and its thickness from `thickness` (or `Eback`). Example, an aluminium capsule:
     ```text
     projectile p
     element Mo
     mass 100
     Ebeam 18.
     thickness 0.02
     layer 1 Al 0.0254
     layer 2 target
     layer 3 Al 0.0254
     ```
   - **`Ebeam`** is the beam energy before layer 1. The energy entering the target is
     calculated and used for the run; the summary prints both, and a table of the stack
     with the energy entering and leaving each layer and the heat left in it.
   - **Heat** in a layer is the beam power it absorbs, as for the target
     (`reactionrates.f90` line 171): heat [kW] = 1000 × Ibeam [A] × (E-in − E-out) [MeV] /
     projectile charge number. Ibeam / (z e) is the number of particles per second and each
     leaves E-in − E-out in the layer. Energy carried away by reaction products is neglected.
   - **Check**: a target behind a layer gives the same results as the bare target hit at the
     degraded energy, and layers behind the target leave its results unchanged
     (`tests/test_layers.py`, and `tests/test_samples.py` with the library).
   - **Layers are pure elements at the density of `constants.RHOTABLE`** (25 °C) unless a
     density is given, and always use the SRIM tables (`p`, `d`, `t`, `h` and `a` in the 92 elements from H to U).
     Compounds and alloys are not available yet.
   - **A beam that stops**: before the target it is an error, because nothing reaches the
     target. In the target or in a later layer it is a warning, and the layers behind are
     marked "not reached by the beam".
   - **Limits**: mean energies only (no straggling, no beam spread from scattering); every
     layer is uniform and crossed along its thickness.
   - **Changed modules**: `inputdeck.py` (keyword, checks, input echo), `rates.py`
     (`setup_stopping`), `state.py` (`Layer`, `target_layer`), `output.py` (layer table).
11. **Isotope production in the layers** (new; for choosing capsule materials):
   - Every layer the beam reaches is also run as a target of its own: the natural element
     at the layer's density, the same beam, current, area and times, between the E-in and
     E-out of the layer table (or, if the beam stops in the layer, from E-in down to the
     1 keV floor of the SRIM table). The equations are those of the target run (manual
     Eq. 4.20-4.28).
   - **Output**: the complete ISOTOPIA output of each layer in the folder
     `layer<n>_<element>` of the output directory, and at the end of the main output a list
     per layer of the radioactive products with their activity at the end of irradiation
     and half-life.
   - **Missing data**: a layer whose element has no abundance file or no cross-section
     library gets a warning in the main output in place of results; the run continues.
   - **Limits**: only the primary beam is followed, at its mean energy. Activation by
     secondary neutrons or other particles from the target or a neighbouring layer is not
     included, and it can matter for capsule choice. Layers are natural elements only.
   - **Check**: the folder of a layer holds the same results as running that layer alone
     (`tests/test_samples.py`).
   - **Changed modules**: `run.py` (`layer_production`), `output.py`
     (`layer_production_out`).
12. **Thin targets keep their products** (changed rule; `fortran_defaults` keeps the old one):
   - The Fortran keeps a product only if a *tabulated* energy with a positive cross section
     lies between Eback and Ebeam (`crosssections.f90` line 203). A thin target can lose
     less energy than the spacing of the table, and then every product was dropped: a
     0.004 cm Mo-100 foil at 15.9 MeV gave 2 products, at 16.2 MeV 30.
   - Now a product is kept if the linearly interpolated cross section is positive somewhere
     in the window, the same interpolation the reaction-rate integral uses
     (`reactionrates.f90` lines 246-252). Rates of products that were already kept do not
     change.
   - **Check**: the production of the windows 16.2 → 15.9 and 15.9 → 15.2 MeV adds up to
     that of 16.2 → 15.2 MeV for every product, to 2 × 10⁻⁵ (`tests/test_samples.py`).
   - **Effect on the samples**: none under `fortran_defaults`. Without it, a product whose
     threshold lies between the last table energy inside the window and the next one
     outside it can now appear with a small rate.
   - **Changed module**: `data.py` (`crosssections`, `_positive_in_window`).
