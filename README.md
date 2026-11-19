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
| `isotopia_py/constants.py` | `constants.f90`, `A0_isotopia_mod.f90` | particles, element symbols, densities, array sizes |
| `isotopia_py/state.py` | `A0_isotopia_mod.f90` | the `State` dataclass holding every run variable |
| `isotopia_py/fortran.py` | – | gfortran number formatting, list-directed reads, `locate`/`pol1`, float32 helpers |
| `isotopia_py/compare.py` | `samples/verify` | numeric comparison of two output directories |
| `isotopia_py/io.py` | – | readers for `.inp` and `.act` files, for post-processing |
| `isotopia_py/decaylib.py` | – (new) | every decay mode, branching ratio and daughter isomeric state from the same decay files |
| `isotopia_py/matrix.py` | – (new) | general Bateman solver N(t) = exp(Mt) N(0): decay network plus reaction terms |
| `isotopia_py/irradiation.py` | – (new) | irradiation + cooling of a target with the matrix solver, rates taken from a port run |
| `isotopia_py/daughters.py` | `reactionrates.f90` (weights) | reactions on products (multiple hits) inside an A/Z window, rates from the target's energy grid and weights |
| `isotopia_py/isotopia_compat.py` | `equations.f90` | ISOTOPIA-compatible mode: the port's equations, defects included, as one matrix exponential (for validation) |

## Setup

Python 3.10+ with NumPy (and SciPy for `matrix.py`). The port needs the Fortran repository (for `files/`
and `samples/`) and the cross-section library. The expected layout is three
sibling directories, the same arrangement ISOTOPIA itself uses:

```text
parent_directory/
├── isotopia/           # github.com/MIDAS-22-033/isotopia_midas, cloned under this name
├── isotopia_python/    # this repository
└── isotopia.libs/      # github.com/arjankoning1/isotopia.libs (about 5 GB)
```

Two environment variables override that layout, exactly as for the Fortran:

```bash
export ISOTOPIA_DIR=/path/to/isotopia          # the Fortran repository (needs files/)
export ISOTOPIA_LIBS=/path/to/isotopia.libs    # optional if it sits next to isotopia/
export ISOTOPIA_USER="Your Name"               # optional, written in output headers
```

Without `ISOTOPIA_DIR` the code looks for an `isotopia` directory next to this one.

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

Hit nuclides deplete at their non-elastic rate, as ISOTOPIA does for the
target. The library has unstable nuclides as targets only if they live about
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
python scripts/run_samples.py                 # all samples vs isotopia/samples/*/org
python scripts/run_samples.py p-Mo100-Tc099m  # one sample
python scripts/run_samples.py --ref runs/fortran   # vs your own Fortran runs
pytest                                        # same, plus unit tests
```

`scripts/run_samples.py` writes the Python results to `runs/python/<case>/` and
reports, per file, whether it is identical to the reference or within tolerance.

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
