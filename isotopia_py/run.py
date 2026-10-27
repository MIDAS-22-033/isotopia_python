"""Main program: isotopia.f90, isotopiainitial.f90, reacinitial.f90,
production.f90 and the isotope loop of natural.f90.
"""
from __future__ import annotations

import sys
from typing import TextIO

from . import data, inputdeck, kinetics, output, rates
from .constants import DAYSEC, HOURSEC, MINUTESEC, PARA, PARZ, RHOTABLE, YEARSEC
from .fortran import S, time_breakdown
from .state import State


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
    if st.rho_target == S(-1):
        st.rho_target = RHOTABLE[st.Ztarget - 1]
    # Irradiation and cooling times, transformed back to years, days, etc.
    st.Tir = _seconds(st.Tirrad, st.unitTirrad, st.timeunit)
    st.Tirrad = [0] + time_breakdown(st.Tir, YEARSEC, DAYSEC, HOURSEC, MINUTESEC, single=True)
    st.Tco = _seconds(st.Tcool, st.unitTcool, st.timeunit)
    st.Tcool = [0] + time_breakdown(st.Tco, YEARSEC, DAYSEC, HOURSEC, MINUTESEC, single=True)
    # Initial compound nucleus
    st.Zcomp = st.Ztarget + PARZ[st.k0]
    st.Acomp = st.Atarget + PARA[st.k0]


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


def run(lines: list[str], out: TextIO | None = None, outdir: str = ".") -> State:
    """Run ISOTOPIA on the given input lines; output files go to ``outdir``."""
    st = State(out=out or sys.stdout, outdir=outdir)
    inputdeck.machine(st)
    inputdeck.isotopiainput(st, lines)
    isotopiainitial(st)
    production(st)
    if st.flagnatural:
        natural(st)
    return st
