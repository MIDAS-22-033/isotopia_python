"""Main program: isotopia.f90, isotopiainitial.f90, reacinitial.f90,
production.f90 and the isotope loop of natural.f90.
"""
from __future__ import annotations

import os
import sys
from typing import TextIO

from . import data, inputdeck, kinetics, output, rates
from .constants import (DAYSEC, HOURSEC, MINUTESEC, PARA, PARZ, RHOTABLE, RHOTABLE_FORTRAN,
                        YEARSEC, allotrope_warning, nuc)
from .fortran import S, time_breakdown
from .state import IsotopiaError, State


def _seconds(times: list[int], units: list[str], timeunit: str):
    """Irradiation or cooling time in seconds (single precision)."""
    T = S(0)
    per = {"y": YEARSEC, "d": DAYSEC, "h": HOURSEC, "m": MINUTESEC}
    for k in range(1, 6):
        if units[k] in per:
            T = T + per[units[k]] * S(times[k])
        elif units[k] == "s":
            T = T + S(times[k])
    if units[1] == " ":
        if timeunit == "s":
            T = S(times[1])
        if timeunit == "h":
            T = S(times[1]) * HOURSEC
        if timeunit == "d":
            T = S(times[1]) * DAYSEC
    return T


def reacinitial(st: State) -> None:
    if st.rho_target == S(-1) and st.fortran_defaults:
        st.rho_target = RHOTABLE_FORTRAN[st.Ztarget - 1]
    elif st.rho_target == S(-1):
        st.rho_target = RHOTABLE[st.Ztarget - 1]
        # Not in the Fortran: point out other allotrope densities (once, not
        # again for each isotope of a natural target).
        if st.iso == 1:
            for line in allotrope_warning(st.Ztarget, st.rho_target):
                st.write(line)
    # Irradiation and cooling times, transformed back to years, days, etc.
    st.Tir = _seconds(st.Tirrad, st.unitTirrad, st.timeunit)
    st.Tirrad = [0] + time_breakdown(st.Tir, YEARSEC, DAYSEC, HOURSEC, MINUTESEC, single=True)
    st.Tco = _seconds(st.Tcool, st.unitTcool, st.timeunit)
    st.Tcool = [0] + time_breakdown(st.Tco, YEARSEC, DAYSEC, HOURSEC, MINUTESEC, single=True)
    # Initial compound nucleus
    st.Zcomp = st.Ztarget + PARZ[st.k0]
    st.Acomp = st.Atarget + PARA[st.k0]
    # Not in the Fortran: stopping-power model and, for a thickness input, Eback
    if st.k0 > 1:
        rates.setup_stopping(st)


def isotopiainitial(st: State) -> None:
    reacinitial(st)
    if st.iso == 1:
        data.decaydata(st)


def production(st: State) -> None:
    rates.reactionrates(st)
    kinetics.equations(st)
    kinetics.activities(st)
    output.prodout(st)


def natural(st: State) -> None:
    """Run each isotope of a natural element, then merge the results."""
    if st.isonum == 1:
        return
    for iso in range(2, st.isonum + 1):
        st.iso = iso
        inputdeck.isotopiainput(st)
        isotopiainitial(st)
        production(st)
    output.natural_output(st)


# Keywords that describe the target itself; a layer's production run replaces them
_TARGET_KEYWORDS = ("element", "mass", "ebeam", "eback", "thickness", "rho", "targetmass",
                    "stopping", "layer", "xsfile")


def layer_production(st: State, lines: list[str]) -> None:
    """Isotope production in the layers around the target (not in the Fortran).

    Each layer the beam reaches is run as an ISOTOPIA target of its own: the natural element
    ('mass 0') at the layer's density, hit by the same beam (projectile, current, area,
    irradiation and cooling times, units) at the mean energy with which the beam enters the
    layer. The energy window is the layer's E-in to E-out of the stack table (keyword Eback),
    or, if the beam stops in the layer, E-in down to the floor of the SRIM table (keyword
    thickness). Physics and equations are those of the target run (manual Eq. 4.20-4.28);
    nothing new is assumed beyond treating each layer independently:

    - the beam entering a layer is mono-energetic at its mean energy (no straggling), as for
      the target;
    - only the primary beam is followed: reactions caused by secondary neutrons or other
      particles from a neighbouring layer or the target are not included.

    Output: the full ISOTOPIA output of each layer (isotopia.out, .act files, ...) in the
    folder ``layer<n>_<element>`` inside the output directory, and a summary at the end of the
    main output (output.layer_production_out). A layer whose data are missing (no abundance
    file or no cross-section library for its isotopes) gets a warning instead of results.

    Parameters
    ----------
    st : State
        The finished target run; ``st.layers`` hold the energies [MeV].
    lines : list of str
        The input lines of the target run.
    """
    common = [ln for ln in lines if (ln.split() or [""])[0].lower() not in _TARGET_KEYWORDS]
    results = []
    for layer in sorted(st.layers, key=lambda la: la.number):
        if not layer.reached:
            continue
        element = nuc(layer.Z).strip()
        folder = f"layer{layer.number}_{element}"
        subdir = os.path.join(st.outdir, folder)
        os.makedirs(subdir, exist_ok=True)
        # The layer as a target. repr() writes the energies with full precision.
        window = (f"thickness {layer.thickness!r}" if layer.E_out == 0.0
                  else f"Eback {layer.E_out!r}")
        deck = common + [f"element {element}", "mass 0", f"Ebeam {layer.E_in!r}", window,
                         f"rho {layer.rho!r}", "stopping srim"]
        sub, message = None, ""
        with open(os.path.join(subdir, "isotopia.out"), "w", encoding="utf-8", newline="\n") as fh:
            try:
                sub = run(deck, out=fh, outdir=subdir)
            except (IsotopiaError, OSError) as exc:
                message = " ".join(str(exc).split())
                fh.write(str(exc) + "\n")
        results.append((layer, folder, sub, message))
    output.layer_production_out(st, results)


def run(lines: list[str], out: TextIO | None = None, outdir: str = ".",
        fortran_defaults: bool = False) -> State:
    """Run ISOTOPIA on the given input lines; output files go to ``outdir``.

    ``fortran_defaults=True`` uses the Fortran's default density table and the Bethe-Bloch
    stopping power, and adds no new output lines, to reproduce the Fortran's outputs.

    If the input has layers, the isotope production in each layer is calculated after the
    target run (``layer_production``).
    """
    st = State(out=out or sys.stdout, outdir=outdir, fortran_defaults=fortran_defaults)
    inputdeck.machine(st)
    inputdeck.isotopiainput(st, lines)
    isotopiainitial(st)
    production(st)
    if st.flagnatural:
        natural(st)
    if st.layers:
        layer_production(st, lines)
    return st
