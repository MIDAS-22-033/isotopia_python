"""Complete decay data for the matrix solver, read from the ENDF-6 decay files.

The literal port (``data.decaydata``, from decaydata.f90) keeps only the
half-life and the type of the *first* decay mode of each nuclide. The general
Bateman solver in ``matrix.py`` needs every decay mode: its type, branching
ratio and the isomeric state of the daughter. This module reads them from the
same library ISOTOPIA ships, ``isotopia/files/decay/zZZZ`` (JEFF-3.1.1 radioactive
decay data, manual Section 5.1), so the matrix and the port share one data source.

ENDF-6 layout read here (File 8, MT 457, "radioactive decay data"):

    HEAD   ZA, AWR, LIS, LISO, NST, NSP      LISO = isomeric state, NST = 1 if stable
    LIST   T1/2, dT1/2, 0, 0, NC, 0           half-life [s], then NC decay energies (skipped)
    LIST   SPI, PAR, 0, 0, 6*NDK, NDK         then NDK x (RTYP, RFS, Q, dQ, BR, dBR)

RTYP is the decay mode. Multi-step modes are written as digits after the
decimal point, e.g. 1.5 = beta-minus followed by neutron emission. RFS is the
isomeric state of the final nucleus and BR its branching ratio.

Nuclides are keyed (Z, A, I) with I = 0 ground state, 1 first isomer, 2 second.
Units: seconds and s^-1.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .constants import nuc
from .fortran import endf_float

Nuclide = tuple[int, int, int]

# Change in (Z, A) for each ENDF-6 RTYP digit. 6 (spontaneous fission) has no
# single daughter; its products are not tracked.
_RTYP_SHIFT = {
    1: (+1, 0),    # beta-minus
    2: (-1, 0),    # electron capture / beta-plus
    3: (0, 0),     # isomeric transition
    4: (-2, -4),   # alpha
    5: (0, -1),    # neutron emission
    7: (-1, -1),   # proton emission
}
_RTYP_NAME = {1: "b-", 2: "ec/b+", 3: "IT", 4: "a", 5: "n", 6: "SF", 7: "p"}
FISSION = 6


class DecayDataError(LookupError):
    """Required decay data is missing from the library."""


def label(nuclide: Nuclide) -> str:
    """Display name of a nuclide key, e.g. (43, 99, 1) -> 'Tc99_m1'."""
    z, a, i = nuclide
    name = f"{nuc(z).strip() if z > 0 else 'n'}{a}"
    return name if i == 0 else f"{name}_m{i}"


@dataclass(frozen=True)
class DecayMode:
    """One decay branch of a nuclide.

    Attributes
    ----------
    rtyp : str
        ENDF-6 decay mode code as written in the file, e.g. '1' or '1.5'.
    daughter : Nuclide or None
        Final nuclide (Z, A, I); None when the products are not tracked
        (spontaneous fission).
    branching : float
        Branching ratio (dimensionless).
    """

    rtyp: str
    daughter: Nuclide | None
    branching: float

    @property
    def name(self) -> str:
        """Readable mode name, e.g. 'b-,n' for RTYP 1.5."""
        return ",".join(_RTYP_NAME.get(int(d), "?" + d) for d in self.rtyp.replace(".", ""))


@dataclass(frozen=True)
class NuclideDecay:
    """Decay properties of one nuclide.

    Attributes
    ----------
    nuclide : Nuclide
        Key (Z, A, I).
    half_life : float
        Half-life [s]; ``math.inf`` for stable nuclides.
    lam : float
        Total decay constant [s^-1]; 0 for stable nuclides.
    modes : tuple of DecayMode
        All decay branches, in file order. Empty for stable nuclides.
    source : str
        Library file the data were read from.
    """

    nuclide: Nuclide
    half_life: float
    lam: float
    modes: tuple[DecayMode, ...]
    source: str


# ----------------------------------------------------------------------------
# ENDF-6 parsing
# ----------------------------------------------------------------------------

def _decode_mode(z: int, a: int, rtyp: float, rfs: float) -> tuple[str, Nuclide | None]:
    """Mode code as text and the daughter it leads to (None if not tracked)."""
    code = f"{rtyp:.5f}".rstrip("0").rstrip(".")
    digits = [int(c) for c in code.replace(".", "")]
    if FISSION in digits:
        return code, None
    if any(d not in _RTYP_SHIFT for d in digits):
        raise DecayDataError(f"unknown ENDF decay mode RTYP={code} for {label((z, a, 0))}")
    for d in digits:
        dz, da = _RTYP_SHIFT[d]
        z, a = z + dz, a + da
    return code, (z, a, int(rfs))


def _list_record(lines: list[str], pos: int) -> tuple[list[float], list[float], int]:
    """Read one ENDF LIST record at ``lines[pos]``: (6 header fields, NPL values, next position)."""
    head = [endf_float(lines[pos][11 * k:11 * k + 11]) for k in range(6)]
    npl = int(head[4])
    nlines = (npl + 5) // 6
    values = [endf_float(line[11 * k:11 * k + 11])
              for line in lines[pos + 1:pos + 1 + nlines] for k in range(6)][:npl]
    return head, values, pos + 1 + nlines


def read_decay_file(path: str | Path) -> dict[Nuclide, NuclideDecay]:
    """Read every nuclide in one zZZZ decay file.

    Parameters
    ----------
    path : str or Path
        ENDF-6 decay file for one element.

    Returns
    -------
    dict
        NuclideDecay per nuclide key (Z, A, I). Half-lives in s, decay
        constants in s^-1.
    """
    # Collect the MF8/MT457 lines of each material (MAT number in columns 67-70).
    sections: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n").ljust(80)
            if line[70:72].strip() == "8" and line[72:75].strip() == "457":
                sections.setdefault(line[66:70], []).append(line)

    out: dict[Nuclide, NuclideDecay] = {}
    for lines in sections.values():
        # HEAD: ZA, AWR, LIS, LISO, NST, NSP
        za, _, _, liso, nst = (endf_float(lines[0][11 * k:11 * k + 11]) for k in range(5))
        z, a = divmod(round(za), 1000)
        key = (z, a, int(liso))
        # LIST 1: half-life; LIST 2: spin, parity and the decay modes
        head, _, pos = _list_record(lines, 1)
        _, modes_raw, _ = _list_record(lines, pos)
        t_half = head[0]
        if int(nst) == 1 or t_half <= 0.0:
            out[key] = NuclideDecay(key, math.inf, 0.0, (), str(path))
            continue
        modes = []
        for k in range(0, len(modes_raw), 6):
            rtyp, rfs, _, _, br, _ = modes_raw[k:k + 6]
            code, daughter = _decode_mode(z, a, rtyp, rfs)
            modes.append(DecayMode(code, daughter, br))
        out[key] = NuclideDecay(key, t_half, math.log(2.0) / t_half, tuple(modes), str(path))
    return out


# ----------------------------------------------------------------------------
# Library access
# ----------------------------------------------------------------------------

def default_decay_dir() -> Path:
    """Decay library directory, found the same way as ``inputdeck.machine``.

    Uses ``$ISOTOPIA_DIR/files/decay``, or ``../isotopia/files/decay`` next to
    this repository when ISOTOPIA_DIR is unset.
    """
    code_dir = os.environ.get("ISOTOPIA_DIR") or Path(__file__).resolve().parents[2] / "isotopia"
    return Path(code_dir) / "files" / "decay"


class DecayLibrary:
    """Decay data for any nuclide, reading each element file once, on first use.

    Parameters
    ----------
    decay_dir : str or Path, optional
        Directory holding the zZZZ files. Default: ``default_decay_dir()``.
    """

    def __init__(self, decay_dir: str | Path | None = None):
        self.decay_dir = Path(decay_dir) if decay_dir is not None else default_decay_dir()
        if not self.decay_dir.is_dir():
            raise DecayDataError(f"decay library directory not found: {self.decay_dir}")
        self._elements: dict[int, dict[Nuclide, NuclideDecay]] = {}

    def file_for(self, z: int) -> Path:
        """Path of the decay file for element Z."""
        return self.decay_dir / f"z{z:03d}"

    def get(self, nuclide: Nuclide) -> NuclideDecay | None:
        """Decay data for (Z, A, I), or None if the library has no entry."""
        z = nuclide[0]
        if z not in self._elements:
            path = self.file_for(z)
            self._elements[z] = read_decay_file(path) if path.is_file() else {}
        return self._elements[z].get(nuclide)
