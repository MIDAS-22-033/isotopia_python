"""Run state: the Python counterpart of the Fortran module A0_isotopia_mod.

ISOTOPIA keeps everything in global arrays dimensioned (Z, A, isomer, time).
Here those become dictionaries keyed by (Z, A, is) that create zero-filled
entries on first access, so values that the Fortran never resets survive from
one isotope run to the next exactly as they do there (this matters for
natural-element targets).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO

import numpy as np

from .constants import NUMISO, NUMTIME
from .fortran import S


class IsotopiaError(RuntimeError):
    """A fatal input or data error; the Fortran prints a message and stops."""


class TimeArrays(dict):
    """Per-nuclide time series, zero until first written (Fortran sgl arrays)."""

    def __missing__(self, key):
        value = np.zeros(NUMTIME + 1, dtype=np.float32)
        self[key] = value
        return value


class Defaulted(dict):
    """A dict that returns (and stores) a default for missing keys."""

    def __init__(self, default):
        super().__init__()
        self._default = default

    def __missing__(self, key):
        value = self._default() if callable(self._default) else self._default
        self[key] = value
        return value

    def reset(self):
        self.clear()


@dataclass
class State:
    out: TextIO = field(default_factory=lambda: sys.stdout)
    outdir: str = "."

    # Input lines
    inline: list[str] = field(default_factory=list)

    # Paths and run info (machine.f90)
    path: str = ""
    crosspath: str = ""
    libname: str = "iaea.2026"
    date: str = ""
    user: str = "Unknown User"
    source: str = "ISOTOPIA"
    oformat: str = "YANDF-0.4"

    # Natural element / isotope counter
    iso: int = 1
    isonum: int = 1
    isotope: list[int] = field(default_factory=lambda: [0] * (NUMISO + 1))
    abun: list = field(default_factory=lambda: [S(0)] * (NUMISO + 1))
    natstring: list[str] = field(default_factory=lambda: ["    "] * (NUMISO + 1))

    # Main input
    flagnatural: bool = False
    flagdecay: bool = True
    flagcross: bool = False
    flagZAoutput: bool = False
    flagselfshield: bool = True
    ptype0: str = " "
    Starget: str = "  "
    Ztarget: int = 0
    Atarget: int = -1
    k0: int = 0
    Zdepth: int = 10
    Adepth: int = 20
    xsfile: dict = field(default_factory=dict)

    # Files
    nuclide: str = ""
    targetnuclide: str = ""
    decaypath: str = ""
    xspath: str = ""

    # Irradiation input
    Ebeam: np.float32 = S(-1)
    Eback: np.float32 = S(-1)
    Ibeam_input: np.float32 = S(1)
    Ibeam: np.float32 = S(0)
    Area: np.float32 = S(1)
    thickness: np.float32 = S(0.05)
    fgamma: np.float32 = S(0.3)
    rho_target: np.float32 = S(-1)
    targetmass: np.float32 = S(0)
    targetmass_input: np.float32 = S(-1)
    fluxtotal: np.float32 = S(1e14)
    timeunit: str = "h"
    radiounit: str = "bq"
    yieldunit: str = "g"
    currentunit: str = "ma"
    Tirrad: list[int] = field(default_factory=lambda: [0] * 6)   # index 1..5
    unitTirrad: list[str] = field(default_factory=lambda: [" "] * 6)
    Tcool: list[int] = field(default_factory=lambda: [0] * 6)
    unitTcool: list[str] = field(default_factory=lambda: [" "] * 6)

    # Conversion
    rstr: str = "Bq"
    ystr: str = "g"
    tstr: str = "h"
    cstr: str = "mA"
    rfac: np.float32 = S(1)
    mfac: np.float32 = S(1)
    tfac: np.float32 = S(1)
    cfac: np.float32 = S(1)

    # Reaction initialisation
    Tir: np.float32 = S(0)
    Tco: np.float32 = S(0)
    Zcomp: int = 0
    Acomp: int = 0

    # Decay data, keyed (Z, A, is)
    Thalf: Defaulted = field(default_factory=lambda: Defaulted(S(1.0e30)))
    lam: Defaulted = field(default_factory=lambda: Defaulted(S(0)))
    rtyp: Defaulted = field(default_factory=lambda: Defaulted(0))
    Td: Defaulted = field(default_factory=lambda: Defaulted(lambda: [0] * 6))

    # Cross sections (crosssections.f90), 1-based arrays with zero padding
    Ein: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    xsrp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    Enon: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    xsnon: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    Etot: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    xstot: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    Nenrp: int = 0
    Nennon: int = 0
    Nentot: int = 0
    rpexist: Defaulted = field(default_factory=lambda: Defaulted(False))
    Nisomer: Defaulted = field(default_factory=lambda: Defaulted(-1))   # keyed (Z, A)
    residual_listing: dict = field(default_factory=dict)

    # Reaction rates
    Leff: np.float32 = S(0)
    V_target: np.float32 = S(0)
    M_target: np.float32 = S(0)
    projnum: np.float32 = S(0)
    heat: np.float32 = S(0)
    Eaverage: np.float32 = S(0)
    selfshield_av: np.float32 = S(1)
    reaction_rate: Defaulted = field(default_factory=lambda: Defaulted(S(0)))
    sacs: Defaulted = field(default_factory=lambda: Defaulted(S(0)))

    # Production
    N_0: np.float32 = S(0)
    N_0_nat: np.float32 = S(0)
    Ntime: int = NUMTIME // 2
    Tgrid: np.ndarray = field(default_factory=lambda: np.zeros(NUMTIME + 1, dtype=np.float32))
    Yexist: Defaulted = field(default_factory=lambda: Defaulted(False))
    Ynatexist: Defaulted = field(default_factory=lambda: Defaulted(False))
    Tmax: Defaulted = field(default_factory=lambda: Defaulted(S(0)))
    Tmaxactivity: Defaulted = field(default_factory=lambda: Defaulted(0))
    Tp: Defaulted = field(default_factory=lambda: Defaulted(lambda: [0] * 6))
    Niso: TimeArrays = field(default_factory=TimeArrays)
    Nisorel: TimeArrays = field(default_factory=TimeArrays)
    Nisotot: TimeArrays = field(default_factory=TimeArrays)          # keyed Z
    activity: TimeArrays = field(default_factory=TimeArrays)
    specactivity: TimeArrays = field(default_factory=TimeArrays)
    yld: TimeArrays = field(default_factory=TimeArrays)
    Nisonat: TimeArrays = field(default_factory=TimeArrays)
    Nisorelnat: TimeArrays = field(default_factory=TimeArrays)
    Nisototnat: TimeArrays = field(default_factory=TimeArrays)       # keyed Z
    Nelrel: TimeArrays = field(default_factory=TimeArrays)           # keyed Z
    activitynat: TimeArrays = field(default_factory=TimeArrays)
    specactivitynat: TimeArrays = field(default_factory=TimeArrays)
    yieldnat: TimeArrays = field(default_factory=TimeArrays)
    reaction_ratenat: Defaulted = field(default_factory=lambda: Defaulted(S(0)))

    def write(self, line: str = "") -> None:
        """Write one record to standard output (Fortran unit *)."""
        self.out.write(line + "\n")
