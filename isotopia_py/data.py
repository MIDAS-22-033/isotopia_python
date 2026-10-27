"""Nuclear data readers: abundance.f90, decaydata.f90 and crosssections.f90."""
from __future__ import annotations

import os

import numpy as np

from . import yandf
from .constants import DAYSEC, HOURSEC, MINUTESEC, NUMISO, PARA, PARSYM, YEARSEC, nuc
from .fortran import S, endf_float, f, fixed_int, i, list_reals, slog, time_breakdown
from .state import IsotopiaError, State


# ----------------------------------------------------------------------------
# abundance.f90
# ----------------------------------------------------------------------------

def abundance(st: State) -> None:
    """Read natural abundances (user file or files/abundance/zNNN) and normalise."""
    abfile = None
    for line in st.inline:
        if line[:9] == "abundance":
            rest = line[10:].strip()
            if rest:
                abfile = rest.replace(",", " ").split()[0]
                break
    if abfile is None:
        abfile = st.path + "files/abundance/z" + i(st.Ztarget, 3, 3)
    if not os.path.exists(abfile):
        raise IsotopiaError(" ISOTOPIA-error: No natural isotopes for this element, "
                            "the mass keyword must be different from 0")
    n = 0
    with open(abfile, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n").ljust(19)
            try:
                ia = fixed_int(line[4:8])
                ab = S(float(line[8:19])) if line[8:19].strip() else S(0)
            except ValueError:
                raise IsotopiaError(f" ISOTOPIA-error: Format error in abundance file {abfile}") from None
            if ia == 0:
                continue
            n += 1
            if n >= NUMISO:
                raise IsotopiaError(f" ISOTOPIA-error: Too many isotopes in abundance file{abfile}"
                                    " Increase numiso in A0_isotopia_mod.f90")
            st.isotope[n] = ia
            st.abun[n] = S(0.01) * ab
    st.isonum = n
    abtot = S(0)
    for k in range(1, n + 1):
        abtot = abtot + st.abun[k]
    for k in range(1, n + 1):
        st.abun[k] = st.abun[k] / abtot
    st.write()
    st.write(" Calculation for multi-isotope case")
    st.write()
    st.write("  Isotope Abundance")
    st.write()
    for k in range(1, n + 1):
        st.write("  " + i(st.isotope[k], 3) + nuc(st.Ztarget) + f(st.abun[k], 11, 6))
    for k in range(1, n + 1):
        st.natstring[k] = "." + i(st.isotope[k], 3, 3)


# ----------------------------------------------------------------------------
# decaydata.f90
# ----------------------------------------------------------------------------

def _lines80(path: str):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            yield raw.rstrip("\r\n")[:80].ljust(80)


def decaydata(st: State) -> None:
    """Half-lives and beta-decay types from the JEFF-3.1.1 decay files."""
    if st.isonum > 1:
        Ahigh = st.isotope[st.isonum] + PARA[st.k0]
    else:
        Ahigh = st.Acomp
    st.lam.reset()
    st.rtyp.reset()
    st.Thalf.reset()
    st.Td.reset()
    ln2 = slog(S(2))
    for iz in range(st.Zcomp + 1, st.Zcomp - st.Zdepth - 2, -1):
        decayfile = st.decaypath + "z" + i(iz, 3, 3)
        if os.path.exists(decayfile):
            _read_decay_file(st, iz, decayfile, Ahigh, ln2)
        for ia in range(Ahigh, st.Acomp - st.Adepth - 1, -1):
            if st.Thalf[(iz, ia, 1)] > S(1e17):
                st.rtyp[(iz, ia, 1)] = st.rtyp[(iz, ia, -1)]
                st.lam[(iz, ia, 1)] = st.lam[(iz, ia, -1)]
                st.Thalf[(iz, ia, 1)] = st.Thalf[(iz, ia, -1)]
                st.Td[(iz, ia, 1)] = list(st.Td[(iz, ia, -1)])


def _read_decay_file(st: State, iz: int, decayfile: str, Ahigh: int, ln2) -> None:
    lines = _lines80(decayfile)
    for string in lines:
        if string[71:80] != "1451    5":
            continue
        ia = fixed_int(string[7:10])
        if ia < st.Acomp - st.Adepth:
            continue
        if ia > Ahigh:
            return
        is_ = -1
        if string[10] == "M":
            is_ = 1
        if string[10] == "N":
            is_ = 2
        for string in lines:
            if string[71:80] == "8457    2":
                break
        else:
            return
        key = (iz, ia, is_)
        st.Thalf[key] = S(endf_float(string[0:11]))
        if st.flagdecay:
            NC = fixed_int(string[44:55])
            iline = 1 + int((NC - 1) / 6)
            try:
                for _ in range(iline):
                    next(lines)
                next(lines)
                string = next(lines)
            except StopIteration:
                return
            st.rtyp[key] = int(endf_float(string[0:11]))
        if st.Thalf[key] == 0:
            st.Thalf[key] = S(1.0e30)
        st.Td[key] = [0] + time_breakdown(st.Thalf[key], YEARSEC, DAYSEC, HOURSEC, MINUTESEC,
                                          single=True)
        if st.Thalf[key] > S(1e17):
            st.lam[key] = S(0)
        else:
            st.lam[key] = ln2 / st.Thalf[key]
        if is_ == -1:
            g = (iz, ia, 0)
            st.rtyp[g] = st.rtyp[key]
            st.lam[g] = st.lam[key]
            st.Thalf[g] = st.Thalf[key]
            st.Td[g] = list(st.Td[key])


# ----------------------------------------------------------------------------
# crosssections.f90
# ----------------------------------------------------------------------------

def _padded(values: list[float]) -> np.ndarray:
    """Fortran-style 1-based single-precision array with zeros at 0 and N+1."""
    arr = np.zeros(len(values) + 2, dtype=np.float32)
    arr[1:len(values) + 1] = values
    return arr


def _read_xs_table(path: str) -> tuple[list[float], list[float]]:
    E, xs = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            string = raw.rstrip("\r\n")[:80]
            if string[:1] == "#":
                continue
            e, x = list_reals(string, 2)
            E.append(e)
            xs.append(x)
    return E, xs


def _write_xs_file(st: State, csfile: str, reaction: str, E, xs, N: int) -> None:
    quantity = "Cross section"
    with open(os.path.join(st.outdir, csfile), "w", encoding="utf-8", newline="\n") as fh:
        topline = st.targetnuclide.rstrip() + reaction.rstrip() + " " + quantity
        yandf.write_header(fh, 0, topline, st.source, st.user, st.date, st.oformat)
        yandf.write_target(fh, 0, st.Ztarget, st.Atarget, st.targetnuclide)
        yandf.write_reaction(fh, 0, reaction, 0.0, 0.0, 3, 3)
        yandf.write_quantity(fh, 0, quantity)
        yandf.write_datablock(fh, 0, 2, N, ["E", "xs"], ["MeV", "mb"])
        for k in range(1, N + 1):
            fh.write(f"{_es(E[k])}{_es(xs[k])}\n")


def _es(x) -> str:
    from .fortran import es
    return es(x, 15, 6)


def _residual_exists(st: State, csfile: str) -> bool:
    """os.path.exists, with one cached directory listing per residual directory."""
    d, name = os.path.split(csfile)
    if d not in st.residual_listing:
        try:
            st.residual_listing[d] = set(os.listdir(d))
        except OSError:
            st.residual_listing[d] = None
    listing = st.residual_listing[d]
    if listing is None:
        return os.path.exists(csfile)
    return name in listing


def crosssections(st: State, iz: int, ia: int, is_: int) -> None:
    """Read residual production, total and non-elastic cross sections."""
    k0 = st.k0
    st.Ein = np.zeros(2, dtype=np.float32)
    st.xsrp = np.zeros(2, dtype=np.float32)
    if iz == 0 and ia == 0 and is_ == -1:
        for typ in (1, 2):
            if k0 != 1 and typ == 2:
                break
            if typ == 2 and not st.flagselfshield:
                break
            st.Etot = np.zeros(2, dtype=np.float32)
            st.xstot = np.zeros(2, dtype=np.float32)
            st.Nentot = 0
            st.Enon = np.zeros(2, dtype=np.float32)
            st.xsnon = np.zeros(2, dtype=np.float32)
            st.Nennon = 0
            if k0 == 1:
                mt = "MT003" if typ == 1 else "MT001"
            else:
                mt = "MT005"
            pfile = st.xspath + "xs/" + PARSYM[k0] + "-" + st.nuclide + "-" + mt + "." + st.libname
            if not os.path.exists(pfile):
                raise IsotopiaError(f" ISOTOPIA-error: Cross section file does not exist: {pfile}")
            E, xs = _read_xs_table(pfile)
            st.Ein = _padded(E)
            st.xsrp = _padded(xs)
            st.Nenrp = len(E)
            if typ == 1:
                st.Enon, st.xsnon, st.Nennon = st.Ein.copy(), st.xsrp.copy(), len(E)
            else:
                st.Etot, st.xstot, st.Nentot = st.Ein.copy(), st.xsrp.copy(), len(E)
            st.rpexist[(0, 0, -1)] = True
            if st.flagcross:
                if typ == 1:
                    reaction = "(" + st.ptype0 + ",non)"
                    csfile = PARSYM[k0] + "-" + st.nuclide + "-non." + st.libname
                    _write_xs_file(st, csfile, reaction, st.Enon, st.xsnon, st.Nennon)
                else:
                    reaction = "(" + st.ptype0 + ",tot)"
                    csfile = PARSYM[k0] + "-" + st.nuclide + "-tot." + st.libname
                    _write_xs_file(st, csfile, reaction, st.Etot, st.xstot, st.Nentot)
        return

    st.Nenrp = 0
    ZAstring = i(iz, 3, 3) + i(ia, 3, 3) + {0: "g", 1: "m", 2: "n"}.get(is_, "")
    reaction = "(" + st.ptype0 + ",x)"
    override = st.xsfile.get((iz, ia, is_), "")
    if override.strip():
        nucfile = "file.loc"
        csfile = override
        exists = os.path.exists(csfile)
    else:
        nucfile = PARSYM[k0] + "-" + st.nuclide + "-rp" + ZAstring + "." + st.libname
        csfile = st.xspath + "residual/" + nucfile
        exists = _residual_exists(st, csfile)
    if not exists:
        return

    flagpositive = False
    Elist: list[float] = []
    xslist: list[float] = []
    E = S(0)
    xs = S(0)
    copy = open(os.path.join(st.outdir, nucfile), "w", encoding="utf-8", newline="\n") \
        if st.flagcross else None
    try:
        with open(csfile, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                string = raw.rstrip("\r\n")[:80]
                if copy:
                    copy.write(string.rstrip() + "\n")
                if string[:1] == "#":
                    continue
                if string.strip():
                    try:
                        e, x = list_reals(string, 2)
                    except ValueError:
                        raise IsotopiaError(
                            f" ISOTOPIA-error: Problem in cross section file {csfile} around"
                            f" E = {_es12(E)} xs = {_es12(xs)}") from None
                    E, xs = S(e), S(x)
                if k0 <= 1 or (st.Eback <= E <= st.Ebeam and xs > 0):
                    flagpositive = True
                if Elist and E < Elist[-1]:
                    raise IsotopiaError(" ISOTOPIA-error: incident energies in cross section file"
                                        f" not in increasing order at E = {_es12(E)}")
                Elist.append(E)
                xslist.append(xs)
    finally:
        if copy:
            copy.close()
    st.Ein = _padded(Elist)
    st.xsrp = _padded(xslist)
    key = (iz, ia, is_)
    if flagpositive:
        st.rpexist[key] = True
        st.Nisomer[(iz, ia)] = max(st.Nisomer[(iz, ia)], is_)
    else:
        st.rpexist[key] = False
    st.Nenrp = len(Elist)

    # Subtract the inelastic cross section from the non-elastic cross section
    if iz == st.Ztarget and ia == st.Atarget and is_ == -1:
        from .fortran import locate, pol1
        N = st.Nenrp
        for k in range(1, st.Nennon + 1):
            Ek = st.Enon[k]
            if Ek < st.Ein[1] or Ek > st.Ein[N]:
                continue
            nen = locate(st.Ein, 1, N, Ek)
            if nen == 0:
                continue
            xsr = pol1(st.Ein[nen], st.Ein[nen + 1], st.xsrp[nen], st.xsrp[nen + 1], Ek)
            st.xsnon[k] = max(st.xsnon[k] - xsr, S(0))
        if st.flagcross:
            csfile = PARSYM[k0] + "-" + st.nuclide + "-non-ninl." + st.libname
            _write_xs_file(st, csfile, "(" + st.ptype0 + ",non)", st.Enon, st.xsnon, st.Nennon)


def _es12(x) -> str:
    from .fortran import es
    return es(x, 12, 5)
