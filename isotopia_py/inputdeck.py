"""Input handling: machine.f90, readinput.f90, convert.f90, getkeywords.f90,
input.f90, checkkeyword.f90, checkvalue.f90, checkfiles.f90, conversion.f90
and inputout.f90.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from . import data
from .constants import NUMA, NUMISOM, NUMLINES, NUMZ, PARSYM, nuc
from .fortran import S, ReadError, a, es, f, i, read_char, read_int, read_real, tab
from .state import IsotopiaError, State

LINELEN = 132


def _fortran_path(p: str) -> str:
    return str(p).replace("\\", "/")


# ----------------------------------------------------------------------------
# machine.f90
# ----------------------------------------------------------------------------

def machine(st: State) -> None:
    """Set data directories, date and user.

    As in the Fortran, ISOTOPIA_DIR points at the isotopia code directory and
    the cross sections default to a sibling ``isotopia.libs``. The Python port
    adds ISOTOPIA_LIBS to point at the library directly, and falls back to an
    ``isotopia`` directory next to this repository when ISOTOPIA_DIR is unset.
    """
    code_dir = os.environ.get("ISOTOPIA_DIR", "")
    if not code_dir:
        guess = Path(__file__).resolve().parents[2] / "isotopia"
        code_dir = str(guess) if guess.is_dir() else "/path/to/isotopia/"
    code_dir = _fortran_path(code_dir)
    if len(code_dir) > 1 and code_dir.endswith("/"):
        code_dir = code_dir[:-1]
    k = code_dir.rfind("/")
    base_dir = code_dir[:k + 1] if k >= 0 else "./"
    st.path = code_dir + "/"
    st.crosspath = base_dir + "isotopia.libs/"
    libs = os.environ.get("ISOTOPIA_LIBS", "")
    if libs:
        libs = _fortran_path(libs)
        st.crosspath = libs if libs.endswith("/") else libs + "/"
    st.libname = "iaea.2026"
    if not os.path.exists(st.path + "files/abundance/z001"):
        raise IsotopiaError(
            "ISOTOPIA error: data files not found.\n"
            f"Expected file: {st.path}files/abundance/z001\n"
            "Set the ISOTOPIA_DIR environment variable to the isotopia directory.")
    st.iso = 1
    st.natstring = ["    "] * len(st.natstring)
    st.isonum = 1
    st.date = _dt.date.today().strftime("%Y-%m-%d")
    st.user = os.environ.get("ISOTOPIA_USER", "") or "Unknown User"


# ----------------------------------------------------------------------------
# readinput.f90, convert.f90, getkeywords.f90
# ----------------------------------------------------------------------------

_CASE_KEYS = ["abundance", "crosspath", "format", "source", "user", "xsfile"]


def convert(line: str) -> str:
    """Lowercase an input line, except what follows a filename/string keyword."""
    low = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in line)
    for k in range(0, 111):
        for kw in reversed(_CASE_KEYS):
            L = len(kw)
            if low[k:k + L] == kw:
                return low[:k + L] + line[k + L:]
    return low


def readinput(st: State, lines: list[str]) -> None:
    """Store the input lines (padded to 132 characters) and lowercase them."""
    if st.iso != 1:
        return
    if len(lines) >= NUMLINES:
        raise IsotopiaError(
            f" ISOTOPIA-error: Number of input lines exceeds{i(NUMLINES, 5)}\n"
            " numlines in A0_isotopia_mod.f90 should be increased")
    st.inline = [convert(ln.rstrip("\r\n")[:LINELEN].ljust(LINELEN)) for ln in lines]


def getkeywords(line: str) -> list[str]:
    """Split a line into words; returns a 1-based list of 40 words ('' = blank)."""
    word = [""] * 41
    prev = " "
    nkey = 0
    ibeg = 0
    for pos, ch in enumerate(line[:LINELEN].ljust(LINELEN)):
        if ch != " " and prev == " ":
            nkey += 1
            ibeg = pos
        if ch == " " and prev != " " and 1 <= nkey <= 40:
            word[nkey] = line[ibeg:pos]
        prev = ch
    return word


def _read_error(line: str, detail: str = "") -> IsotopiaError:
    msg = f" ISOTOPIA-error: Error in {line.rstrip()}"
    if detail:
        msg += f"\n              {detail}"
    return IsotopiaError(msg)


# ----------------------------------------------------------------------------
# input.f90
# ----------------------------------------------------------------------------

def read_keywords(st: State) -> None:
    """Set defaults and read all keywords from the stored input lines."""
    st.flagnatural = False
    st.ptype0 = " "
    st.Ztarget = 0
    st.Starget = "  "
    st.Atarget = -1
    st.Zdepth = 10
    st.Adepth = 20
    st.Ebeam = S(-1)
    st.Eback = S(-1)
    st.Ibeam_input = S(1)
    st.timeunit = "h"
    st.radiounit = "bq"
    st.yieldunit = "g"
    st.currentunit = "ma"
    st.Area = S(1)
    st.fgamma = S(0.3)
    st.thickness = S(0.05)
    st.Tirrad = [0] * 6
    st.unitTirrad = [" "] * 6
    st.Tcool = [0] * 6
    st.unitTcool = [" "] * 6
    st.Tirrad[1] = 1
    st.unitTirrad[1] = "d"
    st.Tcool[1] = 1
    st.unitTcool[1] = "d"
    st.rho_target = S(-1)
    st.targetmass_input = S(-1)
    st.fluxtotal = S(1e14)
    st.flagdecay = True
    st.flagZAoutput = False
    st.flagcross = False
    st.flagselfshield = True
    st.xsfile = {}
    st.source = "ISOTOPIA"
    st.oformat = "YANDF-0.4"
    # The Fortran uses the isomer index 'is' before reading it from the line
    # (see the xsfile keyword below); it starts out undefined there.
    is_prev = -1

    def real(word, current):
        return S(read_real(word, current))

    for line in st.inline:
        word = getkeywords(line)
        key = word[1]
        val = word[2]
        ch = val[:1] or " "
        try:
            if key == "projectile":
                st.ptype0 = ch
            elif key == "element":
                if "0" <= ch <= "9":
                    st.Ztarget = read_int(val, st.Ztarget)
                    if st.Ztarget < 1 or st.Ztarget > NUMZ:
                        raise _read_error(line)
                    st.Starget = nuc(st.Ztarget)
                else:
                    s2 = val[:2].ljust(2)
                    st.Starget = chr(ord(s2[0]) - 32) + s2[1]
            elif key == "mass":
                st.Atarget = read_int(val, st.Atarget)
            elif key == "zdepth":
                st.Zdepth = read_int(val, st.Zdepth)
            elif key == "adepth":
                st.Adepth = read_int(val, st.Adepth)
            elif key == "ebeam":
                st.Ebeam = real(val, st.Ebeam)
            elif key == "ibeam":
                st.Ibeam_input = real(val, st.Ibeam_input)
            elif key == "eback":
                st.Eback = real(val, st.Eback)
            elif key == "area":
                st.Area = real(val, st.Area)
            elif key == "thickness":
                st.thickness = real(val, st.thickness)
            elif key == "fgamma":
                st.fgamma = real(val, st.fgamma)
            elif key == "timeunit":
                st.timeunit = read_char(val, 3, st.timeunit)
            elif key == "radiounit":
                st.radiounit = read_char(val, 3, st.radiounit)
            elif key in ("yieldunit", "massunit"):
                st.yieldunit = read_char(val, 3, st.yieldunit)
            elif key == "currentunit":
                st.currentunit = read_char(val, 3, st.currentunit)
            elif key in ("zaoutput", "selfshield", "outcross", "decay"):
                if ch not in ("y", "n"):
                    raise _read_error(line)
                flag = ch == "y"
                if key == "zaoutput":
                    st.flagZAoutput = flag
                elif key == "selfshield":
                    st.flagselfshield = flag
                elif key == "outcross":
                    st.flagcross = flag
                else:
                    st.flagdecay = flag
            elif key in ("tirrad", "tcool"):
                times = st.Tirrad if key == "tirrad" else st.Tcool
                units = st.unitTirrad if key == "tirrad" else st.unitTcool
                times[1] = 0
                units[1] = " "
                for k in range(1, 6):
                    try:
                        T = read_real(word[2 * k], 0.0)
                    except ReadError:
                        break
                    times[k] = int(S(T))
                    units[k] = (word[2 * k + 1][:1] or " ")
            elif key == "rho":
                st.rho_target = real(val, st.rho_target)
            elif key == "targetmass":
                st.targetmass_input = real(val, st.targetmass_input)
            elif key == "fluxtotal":
                st.fluxtotal = real(val, st.fluxtotal)
            elif key == "crosspath":
                # All non-blank characters from column 11 on, written over the
                # start of the existing crosspath (the rest is not cleared).
                new = "".join(c for c in line[10:LINELEN] if c != " ")
                st.crosspath = new + st.crosspath[len(new):]
            elif key == "xsfile":
                iz = read_int(word[2])
                ia = read_int(word[3])
                if iz < 0 or iz > NUMZ or ia < 0 or ia > NUMA:
                    raise IsotopiaError(
                        f" ISOTOPIA-error: Z, A index out of range {line.rstrip()}")
                st.xsfile[(iz, ia, is_prev)] = word[4]
                is_prev = -1
                try:
                    is_prev = read_int(word[5], is_prev)
                except ReadError:
                    continue
                if is_prev < -1 or is_prev > NUMISOM:
                    raise _read_error(line)
            elif key == "source":
                st.source = line[line.find("source") + 7:LINELEN].strip()
            elif key == "user":
                st.user = line[line.find("user") + 5:LINELEN].strip()
            elif key == "format":
                st.oformat = line[line.find("format") + 7:LINELEN].strip()
        except ReadError as exc:
            raise _read_error(line, str(exc)) from None


# ----------------------------------------------------------------------------
# checkkeyword.f90
# ----------------------------------------------------------------------------

_KEYWORDS = {
    "", "abundance", "adepth", "area", "crosspath", "currentunit", "decay", "eback", "ebeam",
    "element", "fgamma", "fluxtotal", "format", "ibeam", "mass", "massunit", "outcross",
    "projectile", "radiounit", "rho", "selfshield", "source", "targetmass", "thickness",
    "timeunit", "tirrad", "tcool", "user", "xsfile", "yieldunit", "zaoutput", "zdepth",
}


def checkkeyword(st: State) -> None:
    for line in st.inline:
        key = getkeywords(line)[1]
        if key[:1] == "#":
            continue
        if key not in _KEYWORDS:
            raise IsotopiaError(f"\n ISOTOPIA-error: Wrong keyword: {a(key, 20)}")


# ----------------------------------------------------------------------------
# checkvalue.f90
# ----------------------------------------------------------------------------

def checkvalue(st: State) -> None:
    def stop(msg):
        raise IsotopiaError(f" ISOTOPIA-error: {msg}")

    if st.ptype0 == " ":
        stop("projectile must be given")
    if st.ptype0 not in PARSYM:
        stop(f"Wrong symbol for projectile: {st.ptype0}")
    st.k0 = PARSYM.index(st.ptype0)
    if st.Ztarget == 0 and st.Starget == "  ":
        stop("element must be given")
    if st.Ztarget == 0:
        for iz in range(1, NUMZ + 1):
            if nuc(iz) == st.Starget:
                st.Ztarget = iz
                break
        else:
            stop(f"Wrong symbol for element: {st.Starget}")
    if st.Atarget == -1:
        stop("target mass must be given")
    if st.Atarget == 0:
        st.flagnatural = True
        if st.iso == 1:
            data.abundance(st)
        st.Atarget = st.isotope[st.iso]
    if st.Atarget <= 5 or st.Atarget > NUMA:
        stop(f"5 < Target mass <= {i(NUMA, 3)}")
    if st.k0 != 1:
        if st.Ebeam == S(-1):
            stop("incident energy must be given")
        if st.Ebeam <= 0 or st.Ebeam > 250:
            stop("0 < Ebeam < 250 MeV")
        if st.k0 > 1:
            if st.Eback == S(-1):
                st.Eback = max(st.Ebeam - S(5), S(0.1))
            elif st.Eback <= 0 or st.Eback > 250:
                stop("0 < Eback < 250 MeV")
            if st.Eback >= st.Ebeam:
                stop("Ebeam must be larger than Eback")
    st.Zdepth = min(st.Zdepth, st.Ztarget - 1)
    st.Adepth = min(st.Adepth, st.Atarget - 1)
    if st.Zdepth < 0:
        stop("Zdepth >= 0")
    if st.Adepth < 0:
        stop(" Adepth >= 0")
    if st.Ibeam_input <= 0 or st.Ibeam_input > 10000:
        stop("0 < = Ibeam < 10000 in selected currentunit")
    if st.Area <= 0 or st.Area > 10000:
        stop("0 <  Area <= 10000 cm^2")
    if st.thickness <= 0 or st.thickness > 10000:
        stop("0 <  thickness <= 10000 cm")
    for k in range(1, 6):
        if st.Tirrad[k] < 0 or st.Tirrad[k] > 1000000:
            stop("0 < = Tirrad <= 1.e6")
        if st.Tcool[k] < 0 or st.Tcool[k] > 1000000:
            stop("0 < = Tcool <= 1.e6")
    for k in range(1, 6):
        if st.unitTirrad[k] not in (" ", "y", "d", "h", "m", "s"):
            stop(f"wrong unit for Tirrad = {i(st.Tirrad[k], 9)}")
        if st.unitTcool[k] not in (" ", "y", "d", "h", "m", "s"):
            stop(f"wrong unit for Tcool = {i(st.Tcool[k], 9)}")
    if st.rho_target != S(-1) and (st.rho_target <= 0 or st.rho_target > 100):
        stop("0 < rhotarget <= 100.")
    if st.targetmass_input != S(-1) and (st.targetmass_input <= 0 or st.targetmass_input > S(1e9)):
        stop("0 < targetmass <= 1.e9")
    if st.fluxtotal <= 0 or st.fluxtotal > S(1e38):
        stop("0 < fluxtotal <= 1.e38")
    if st.fgamma <= 0 or st.fgamma > 1:
        stop("0 < fgamma <= 1.")
    if st.yieldunit not in ("mug", "mg", "g", "kg"):
        stop("yieldunit must be equal to mug, mg, g or kg")
    if st.radiounit not in ("bq", "kbq", "mbq", "gbq", "ci", "kci", "mci"):
        stop("radiounit must be equal to bq, kbq, mbq, gbq, ci, kci or mci")
    if st.timeunit not in ("s", "h", "d"):
        stop("timeunit must be equal to s, h or d")
    if st.currentunit not in ("mua", "ma", "a"):
        stop("currentunit must be equal to mua, ma or a")


# ----------------------------------------------------------------------------
# checkfiles.f90
# ----------------------------------------------------------------------------

def checkfiles(st: State) -> None:
    st.nuclide = st.Starget.strip() + i(st.Atarget, 3, 3)
    st.targetnuclide = st.Starget.strip() + str(st.Atarget)
    nucpath = PARSYM[st.k0] + "/" + st.nuclide + "/"
    st.xspath = st.crosspath.rstrip() + nucpath + st.libname + "/tables/"
    mt = "MT003" if st.k0 == 1 else "MT005"
    pfile = st.xspath + "xs/" + PARSYM[st.k0] + "-" + st.nuclide + "-" + mt + "." + st.libname
    if not os.path.exists(pfile):
        raise IsotopiaError(f" ISOTOPIA-error: Cross section file does not exist: {pfile}")
    st.decaypath = st.path + "files/decay/"
    decayfile = st.decaypath + "z" + i(st.Ztarget, 3, 3)
    if not os.path.exists(decayfile):
        raise IsotopiaError(f" ISOTOPIA-error: Decay data file does not exist: {decayfile}")


# ----------------------------------------------------------------------------
# conversion.f90
# ----------------------------------------------------------------------------

_RADIO = {"bq": ("Bq", 1.0), "kbq": ("kBq", 1.0e-3), "mbq": ("MBq", 1.0e-6),
          "gbq": ("GBq", 1.0e-9), "ci": ("Ci", None), "kci": ("kCi", None), "mci": ("mCi", None)}


def conversion(st: State) -> None:
    st.rstr, st.ystr, st.cstr, st.tstr = "Bq", "g", "mA", "h"
    st.rfac = st.mfac = st.tfac = st.cfac = S(1)
    if st.radiounit in _RADIO:
        st.rstr, fac = _RADIO[st.radiounit]
        if fac is not None:
            st.rfac = S(fac)
        elif st.radiounit == "ci":
            st.rfac = S(1) / S(3.7e10)
        elif st.radiounit == "kci":
            st.rfac = S(1) / S(3.7e13)
        else:
            st.rfac = S(1) / S(3.7e7)
    st.ystr, st.mfac = {"g": ("g", S(1)), "mg": ("mg", S(1e3)), "mug": ("mug", S(1e6)),
                        "kg": ("kg", S(1e-3))}.get(st.yieldunit, (st.ystr, st.mfac))
    st.tstr, st.tfac = {"s": ("s", S(1)), "h": ("h", S(3600)),
                        "d": ("d", S(86400))}.get(st.timeunit, (st.tstr, st.tfac))
    st.cstr, st.cfac = {"ma": ("mA", S(1e-3)), "mua": ("muA", S(1e-6)),
                        "a": ("A", S(1))}.get(st.currentunit, (st.cstr, st.cfac))
    st.Ibeam = st.Ibeam_input * st.cfac
    if st.targetmass_input != S(-1):
        st.targetmass = st.targetmass_input / st.mfac


# ----------------------------------------------------------------------------
# inputout.f90
# ----------------------------------------------------------------------------

def yesno(flag: bool) -> str:
    return "y" if flag else "n"


def inputout(st: State) -> None:
    w = st.write
    w()
    w("    ISOTOPIA-2.23   (Version: July 9, 2026)")
    w()
    w(" " * 10 + " Simulation of medical isotope production")
    w()
    w(" Copyright (C) 2026  A.J. Koning")
    w()
    w(" User: " + st.user.rstrip())
    w(" Date: " + st.date.rstrip())
    w()
    w(" ########## USER INPUT ##########")
    w()
    w(" USER INPUT FILE")
    w()
    for line in st.inline:
        w(" " + line.rstrip())
    w()
    w(" USER INPUT FILE + DEFAULTS")
    w()
    w(" Keyword           Value   Variable     Explanation")
    w()
    w(tab(" user      " + st.user.rstrip(), 28, "user         user for this calculation"))
    w(tab(" source    " + st.source.rstrip(), 28, "source       source for this calculation"))
    w(tab(" format    " + st.oformat.rstrip(), 28, "oformat      format for output"))
    w(" crosspath " + st.crosspath.rstrip() + "   crosspath    directory containing cross sections ")
    w(" #")
    w(" # Nuclear reaction")
    w(" #")
    w(" projectile          " + a(st.ptype0, 1) + "     ptype0       type of incident particle")
    w(" element            " + a(st.Starget, 2) + "     Starget      symbol of target nucleus")
    w(" mass              " + i(st.Atarget, 3) + "     Atarget      mass number of target nucleus")
    if st.k0 != 1:
        w(" #")
        w(" # Accelerator ")
        w(" #")
        w(" Ebeam             " + f(st.Ebeam, 7, 3) + " Ebeam        incident energy in MeV")
        if st.k0 > 1:
            w(" Eback             " + f(st.Eback, 7, 3) + " Eback        lower end of energy range in MeV")
        w(" Ibeam             " + f(st.Ibeam_input, 7, 3) + " Ibeam_input  beam current in " + st.cstr.rstrip())
    w(" fgamma            " + f(st.fgamma, 7, 3) + " fgamma       electron-to-photon conversion effiency")
    w(" #")
    w(" # Output ")
    w(" #")
    w(" radiounit             " + a(st.radiounit, 3) + " radiounit    unit for radioactivity")
    w(" yieldunit (massunit)  " + a(st.yieldunit, 3) + " yieldunit    unit for isotope yield")
    w(" ZAoutput              " + yesno(st.flagZAoutput)
      + "   flagZAoutput output files per Z, A instead of El, A")
    w(" #")
    w(" # Target")
    w(" #")
    for k in range(1, 6):
        if st.Tirrad[k] > 0:
            w(" Tirrad      " + i(st.Tirrad[k], 9) + "     Tirrad       " + a(st.unitTirrad[k], 1)
              + " irradiation time")
    w(" Area              " + f(st.Area, 7, 3) + " Area         target area in cm^2")
    for k in range(1, 6):
        if st.Tcool[k] > 0:
            w(" Tcool       " + i(st.Tcool[k], 9) + "     Tcool        " + a(st.unitTcool[k], 1)
              + " cooling time")
    if st.rho_target != S(-1):
        w(" rho               " + f(st.rho_target, 7, 3) + " rho_target   target density [g/cm^3] ")
    if st.targetmass_input != S(-1):
        w(" targetmass    " + f(st.targetmass_input, 7, 3) + " targetmass_input target mass ")
    if st.k0 == 1:
        w(" fluxtotal " + es(st.fluxtotal, 15, 6) + " fluxtotal    total flux")
    w(" selfshield            " + yesno(st.flagselfshield) + " flagselfshield flag for self-shielding ")
    w(" #")
    w(" # Cross section and decay data")
    w(" #")
    w(" Zdepth            " + i(st.Zdepth, 3)
      + "     Zdepth       depth to which Z numbers are scanned for cross sections")
    w(" Adepth            " + i(st.Adepth, 3)
      + "     Adepth       depth to which A numbers are scanned for cross sections")
    w(" decay               " + yesno(st.flagdecay)
      + "     flagdecay    flag to include decay of nuclides (n only for diagnosis) ")
    w()


def isotopiainput(st: State, lines: list[str] | None = None) -> None:
    """isotopiainput.f90: read, check and echo the input."""
    readinput(st, lines or [])
    read_keywords(st)
    checkkeyword(st)
    checkvalue(st)
    checkfiles(st)
    conversion(st)
    inputout(st)

