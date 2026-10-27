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

## Setup

Python 3.10+ with NumPy. Two environment variables tell the code where the data is,
exactly as for the Fortran:

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

## Known quirks of the Fortran, reproduced here on purpose

These are kept so the outputs match; each is marked in the code and can be
fixed in a separate change:

1. `input.f90`: the `xsfile` keyword stores its file under the isomer index
   read from the *previous* `xsfile` line (the index is read after it is used).
2. For natural targets the per-nuclide arrays (`Niso`, `activity`, ...) are not
   cleared between isotope runs, only overwritten for nuclides that exist in
   the current run. Values for a nuclide that was produced from one isotope but
   not the next survive into the later run's output. The `State` containers
   deliberately persist in the same way.
