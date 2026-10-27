"""Helpers that reproduce Fortran behaviour the port depends on.

ISOTOPIA stores most quantities as single-precision reals (``real(sgl)``)
and does part of its arithmetic in single precision. To reproduce its output
digit for digit, the port keeps those quantities as ``numpy.float32`` (alias
``S``) and converts with ``float()`` wherever Fortran promotes to double.

The formatting functions mimic gfortran edit descriptors (ESw.d, Fw.d, Iw,
Iw.m), including the asterisk fill on overflow.
"""
from __future__ import annotations

import math
import re

import numpy as np

S = np.float32  # Fortran real(sgl)


def sgl(x) -> np.float32:
    """Round a value to single precision, as a Fortran sgl assignment does."""
    return np.float32(x)


# Single-precision intrinsics. glibc's logf/expf/powf are correctly rounded in
# practice, which numpy's float32 ufuncs are not; computing in double and
# rounding reproduces gfortran on Linux.

def slog(x) -> np.float32:
    x = float(x)
    if x <= 0.0:
        return S(-math.inf) if x == 0.0 else S(math.nan)
    return S(math.log(x))


def sexp(x) -> np.float32:
    try:
        return S(math.exp(float(x)))
    except OverflowError:
        return S(math.inf)


def spow(x, y) -> np.float32:
    return S(float(x) ** float(y))


def ssqrt(x) -> np.float32:
    return np.sqrt(S(x))


# ----------------------------------------------------------------------------
# Output formatting (gfortran edit descriptors)
# ----------------------------------------------------------------------------

def es(x, w: int, d: int) -> str:
    """Fortran ESw.d edit descriptor."""
    x = float(x)
    if math.isnan(x):
        return "NaN".rjust(w)
    if math.isinf(x):
        s = "Infinity" if x > 0 else "-Infinity"
        return s.rjust(w) if len(s) <= w else ("Inf" if x > 0 else "-Inf").rjust(w)
    s = f"{x:.{d}E}"
    mant, exp = s.split("E")
    e = int(exp)
    if abs(e) > 99:
        # Without an explicit exponent width, Fortran drops the 'E' for
        # three-digit exponents, e.g. 1.000000+100.
        s = f"{mant}{e:+04d}"
    else:
        s = f"{mant}E{e:+03d}"
    if len(s) > w:
        return "*" * w
    return s.rjust(w)


def f(x, w: int, d: int) -> str:
    """Fortran Fw.d edit descriptor."""
    x = float(x)
    if math.isnan(x):
        return "NaN".rjust(w)
    if math.isinf(x):
        return ("Inf" if x > 0 else "-Inf").rjust(w)
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]  # gfortran prints rounded negative zero without the sign
    if len(s) > w:
        # gfortran drops an optional leading zero before asterisk fill.
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    if len(s) > w:
        return "*" * w
    return s.rjust(w)


def i(n: int, w: int, m: int | None = None) -> str:
    """Fortran Iw or Iw.m edit descriptor."""
    n = int(n)
    if m is None:
        s = str(n)
    else:
        s = f"{abs(n):0{m}d}"
        if n < 0:
            s = "-" + s
    if len(s) > w:
        return "*" * w
    return s.rjust(w)


def a(s: str, w: int) -> str:
    """Fortran Aw edit descriptor for output: truncate or left-justify."""
    return s[:w].ljust(w)


def tab(record: str, col: int, text: str) -> str:
    """Fortran Tn positioning followed by writing ``text`` at column ``col``.

    Tn is absolute, so it may move left and overwrite what was written.
    """
    pos = col - 1
    record = record.ljust(pos)
    return record[:pos] + text + record[pos + len(text):]


def time_breakdown(T, yearsec, daysec, hoursec, minutesec, single: bool) -> list[int]:
    """Split a time in seconds into years, days, hours, minutes, seconds.

    The Fortran repeats ``TT = TT - n * unitsec`` in single or double
    precision depending on how TT is declared; ``single`` selects which.
    """
    # The unit lengths are single precision, so n * unitsec is a single-
    # precision product even when TT itself is double (verified against the
    # Fortran output).
    units = (S(yearsec), S(daysec), S(hoursec), S(minutesec))
    TT = S(T) if single else float(T)
    out = []
    for u in units:
        n = _trunc(TT / (u if single else float(u)))
        out.append(n)
        prod = S(n) * u
        TT = TT - (prod if single else float(prod))
    out.append(_trunc(TT))
    return out


def _trunc(x) -> int:
    """Fortran INT(): truncation toward zero, saturating like gfortran on overflow."""
    x = float(x)
    if math.isnan(x):
        return -2147483648
    if x >= 2147483647.0:
        return 2147483647
    if x <= -2147483648.0:
        return -2147483648
    return int(x)


def fexp(x: float) -> float:
    """Double-precision exp that returns inf on overflow, as Fortran does."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def time_string(T, parts: list[int]) -> str:
    """Format '(es15.6," (",i6, " years ", i3, " days", i3, " hours", ...)'."""
    return (f"{es(T, 15, 6)} ({i(parts[0], 6)} years {i(parts[1], 3)} days"
            f"{i(parts[2], 3)} hours{i(parts[3], 3)} minutes{i(parts[4], 3)} seconds)")


# ----------------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------------

_REAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdDqQ][+-]?\d+|[+-]\d+)?$")
_INT = re.compile(r"^[+-]?\d+$")


class ReadError(ValueError):
    """A Fortran list-directed read failed (istat /= 0)."""


def _token(word: str) -> str | None:
    """First list-directed item of a word; None means a null value."""
    if word == "":
        raise ReadError("end of file")
    tok = re.split(r"[,/]", word, maxsplit=1)[0]
    return tok if tok != "" else None


def read_real(word: str, current=None):
    """Fortran ``read(word, *) x`` for a real; keeps ``current`` on a null value."""
    tok = _token(word)
    if tok is None:
        return current
    if not _REAL.match(tok):
        raise ReadError(f"bad real {tok!r}")
    return float(_to_python_float(tok))


def read_int(word: str, current=None):
    """Fortran ``read(word, *) n`` for an integer."""
    tok = _token(word)
    if tok is None:
        return current
    if not _INT.match(tok):
        raise ReadError(f"bad integer {tok!r}")
    return int(tok)


def read_char(word: str, length: int, current=None):
    """Fortran list-directed read into ``character(len=length)``."""
    tok = _token(word)
    if tok is None:
        return current
    return tok[:length]


def _to_python_float(tok: str) -> float:
    t = tok.replace("d", "e").replace("D", "e").replace("q", "e").replace("Q", "e")
    m = re.match(r"^([+-]?[\d.]+)([+-]\d+)$", t)
    if m:
        t = f"{m.group(1)}e{m.group(2)}"
    return float(t)


def endf_float(field: str) -> float:
    """Read an ENDF-style number such as ' 1.10000-07' (Fortran E11.5 input)."""
    s = field.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r"^([+-]?[\d.]+)([+-]\d+)$", s)
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    raise ValueError(f"cannot read ENDF number {field!r}")


def fixed_int(field: str) -> int:
    """Fortran Iw input: blanks are zero."""
    s = field.strip()
    return int(s) if s else 0


def list_reals(line: str, n: int) -> list[float]:
    """Fortran ``read(line, *) x1, ..., xn`` for reals."""
    toks = line.replace(",", " ").split()
    if len(toks) < n:
        raise ReadError(f"expected {n} values in {line!r}")
    out = []
    for t in toks[:n]:
        if not _REAL.match(t):
            raise ReadError(f"bad real {t!r}")
        out.append(_to_python_float(t))
    return out


# ----------------------------------------------------------------------------
# Numerical routines (locate.f90, pol1.f90)
# ----------------------------------------------------------------------------

def locate(xx, ib: int, ie: int, x) -> int:
    """Find j such that xx(j) <= x < xx(j+1); a literal port of locate.f90.

    ``xx`` is indexed like the Fortran array (index 0 is a real element).
    """
    if ib > ie:
        return 0
    jl = ib - 1
    ju = ie + 1
    ascend = xx[ie] >= xx[ib]
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascend == (x >= xx[jm]):
            jl = jm
        else:
            ju = jm
    if x == xx[ib]:
        return ib
    if x == xx[ie]:
        return ie - 1
    return jl


def pol1(x1, x2, y1, y2, x):
    """First-order interpolation in single precision (pol1.f90)."""
    x1, x2, y1, y2, x = S(x1), S(x2), S(y1), S(y2), S(x)
    with np.errstate(all="ignore"):
        fac = (x - x1) / (x2 - x1)
        return y1 + fac * (y2 - y1)
