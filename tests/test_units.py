"""Unit tests for pieces that can be checked without the cross-section library."""
from __future__ import annotations

import io

import numpy as np
import pytest

from isotopia_py import fortran as F
from isotopia_py.constants import ALLOTROPES, NUC, NUMZ, RHOTABLE, RHOTABLE_FORTRAN
from isotopia_py.inputdeck import checkvalue, convert, getkeywords, read_keywords, readinput
from isotopia_py.kinetics import equations
from isotopia_py.run import reacinitial
from isotopia_py.state import IsotopiaError, State

S = np.float32


def test_density_table():
    assert NUMZ == 118 and NUC[-1] == "Og"
    assert len(NUC) == len(RHOTABLE) == len(RHOTABLE_FORTRAN) == NUMZ
    assert RHOTABLE[10 - 1] == S(0.0008248)    # neon, ideal gas at 25 degC
    assert RHOTABLE[6 - 1] == S(2.2)           # graphite, CRC at 25 degC
    assert RHOTABLE[92 - 1] == S(19.1)         # uranium, last element in the data table
    unknown = [z for z in range(1, NUMZ + 1) if RHOTABLE[z - 1] is None]
    assert unknown == list(range(100, 119))     # Fm-Og: no measured density
    assert RHOTABLE[85 - 1] == S(8.95) and RHOTABLE[87 - 1] == S(2.458)   # At, Fr (Arblaster 2018)
    # Np-Es: CRC density table, except Am-243, Bk-247 and Es-252 from lattice parameters
    assert RHOTABLE[92:99] == [S(r) for r in (20.2, 19.7, 13.79, 13.51, 14.67, 15.1, 8.81)]
    assert ALLOTROPES[97] == (("alpha, dhcp", S(14.67)), ("beta, fcc (metastable)", S(13.15)))  # both Bk-247
    for z, forms in ALLOTROPES.items():        # default allotrope is listed first
        assert forms[0][1] == RHOTABLE[z - 1]


def _check_input(*lines):
    st = State(out=io.StringIO())
    readinput(st, list(lines))
    read_keywords(st)
    checkvalue(st)
    return st


@pytest.mark.parametrize("element", ["119", "124"])
def test_element_beyond_oganesson_is_an_error(element):
    with pytest.raises(IsotopiaError, match="Z must be <= 118"):
        _check_input("projectile n", f"element {element}", "mass 300")


@pytest.mark.parametrize("symbol", ["B9", "C4"])   # the old labels for Z = 119, 124
def test_old_superheavy_labels_are_not_elements(symbol):
    with pytest.raises(IsotopiaError, match="Wrong symbol for element"):
        _check_input("projectile n", f"element {symbol}", "mass 300")


@pytest.mark.parametrize("element,mass", [("Fm", 257), ("118", 294)])
def test_unknown_density_requires_rho(element, mass):
    with pytest.raises(IsotopiaError, match="no verified density.*keyword rho"):
        _check_input("projectile n", f"element {element}", f"mass {mass}")
    st = _check_input("projectile n", f"element {element}", f"mass {mass}", "rho 12.")
    assert st.rho_target == S(12.0)


@pytest.mark.parametrize("element,mass", [("Np", 237), ("Cf", 249), ("Es", 252)])
def test_transuranium_with_density_needs_no_rho(element, mass):
    assert _check_input("projectile n", f"element {element}", f"mass {mass}").rho_target == S(-1)


def _reacinitial_output(z: int, rho: float = -1.0, iso: int = 1):
    st = State(out=io.StringIO())
    st.Ztarget, st.Atarget, st.k0 = z, 2 * z, 2
    st.Ebeam, st.Eback = S(16.0), S(12.0)      # reacinitial now also sets up the stopping power
    st.rho_target = S(rho)
    st.iso = iso
    st.Tirrad, st.unitTirrad = [0, 1, 0, 0, 0, 0], [" ", "d", " ", " ", " ", " "]
    st.Tcool, st.unitTcool = [0, 1, 0, 0, 0, 0], [" ", "d", " ", " ", " ", " "]
    reacinitial(st)
    return st.rho_target, st.out.getvalue()


@pytest.mark.parametrize("z", sorted(ALLOTROPES))
def test_allotrope_message_keeps_table_default(z):
    rho, text = _reacinitial_output(z)
    assert rho == RHOTABLE[z - 1]
    assert "ISOTOPIA-warning: other densities were found" in text
    for form, _ in ALLOTROPES[z]:
        assert form in text


def test_no_allotrope_message_when_rho_given_or_single_form():
    assert _reacinitial_output(6, rho=3.513) == (S(3.513), "")   # user chose diamond
    assert _reacinitial_output(42)[1] == ""                       # Mo: one form
    assert _reacinitial_output(6, iso=2)[1] == ""                 # later isotope of a natural target


def test_es_format_matches_gfortran():
    assert F.es(1.0, 15, 6) == "   1.000000E+00"
    assert F.es(-2.5e-7, 13, 6) == "-2.500000E-07"
    assert F.es(1e100, 15, 6) == "   1.000000+100"
    assert F.es(0.0, 15, 6) == "   0.000000E+00"


def test_f_and_i_formats():
    assert F.f(16.0, 7, 3) == " 16.000"
    assert F.f(0.99999, 8, 5) == " 0.99999"
    assert F.i(7, 3, 3) == "007"
    assert F.i(123456, 3) == "***"


def test_time_breakdown_single_and_double():
    units = (F.S(365.25 * 86400), F.S(86400), F.S(3600), F.S(60))
    secs = 365.25 * 86400 + 2 * 86400 + 3 * 3600 + 4 * 60 + 5
    assert F.time_breakdown(secs, *units, single=False) == [1, 2, 3, 4, 5]
    # float32 cannot hold seconds next to a year (spacing is 2-4 s there)
    assert F.time_breakdown(secs, *units, single=True)[:4] == [1, 2, 3, 4]
    assert F.time_breakdown(2 * 86400 + 3 * 3600 + 4 * 60 + 5, *units, single=True) == [0, 2, 3, 4, 5]


def test_endf_numbers():
    assert F.endf_float(" 1.10000-07") == pytest.approx(1.1e-7)
    assert F.endf_float("4.30850+04") == pytest.approx(43085.0)
    assert F.endf_float("           ") == 0.0


def test_locate_and_pol1():
    xx = np.array([0.0, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)  # index 0 unused
    assert F.locate(xx, 1, 4, 3.0) == 2
    assert F.locate(xx, 1, 4, 1.0) == 1
    assert F.locate(xx, 1, 4, 8.0) == 3
    assert F.pol1(2.0, 4.0, 10.0, 20.0, 3.0) == S(15.0)


def test_input_line_handling():
    assert convert("Element MO") == "element mo"
    # strings after filename keywords keep their case
    assert convert("Xsfile 43 99 /Data/MyFile.txt") == "xsfile 43 99 /Data/MyFile.txt"
    words = getkeywords("tirrad 2 d 3 h")
    assert words[1:6] == ["tirrad", "2", "d", "3", "h"]


def test_direct_production_matches_analytic_solution():
    """A single product with no parents follows N_0 R_i (e^-R_T t - e^-lam t)/(lam - R_T)."""
    st = State(out=io.StringIO())
    st.Ztarget, st.Atarget, st.Zcomp, st.Acomp = 42, 100, 43, 101
    st.Zdepth, st.Adepth = 2, 2    # Z 44..42, A 101..99
    st.k0 = 2
    st.M_target = S(0.2)
    st.Tir, st.Tco = S(86400.0), S(0.0)
    st.abun[1] = S(1.0)
    st.reaction_rate[(0, 0, -1)] = S(1e-9)
    st.reaction_rate[(43, 99, -1)] = S(2e-10)
    st.lam[(43, 99, -1)] = S(3.2e-5)
    st.rpexist[(43, 99, -1)] = True
    st.rpexist[(42, 100, -1)] = True   # the target itself (non-elastic file exists)
    st.flagdecay = False
    equations(st)
    N0 = float(st.N_0)
    t = float(st.Tgrid[st.Ntime])
    expected = N0 * 2e-10 * (np.exp(-1e-9 * t) - np.exp(-3.2e-5 * t)) / (3.2e-5 - 1e-9)
    assert float(st.Niso[(43, 99, -1)][st.Ntime]) == pytest.approx(expected, rel=1e-6)
    assert float(st.Niso[(42, 100, -1)][st.Ntime]) == pytest.approx(N0 * np.exp(-1e-9 * t), rel=1e-6)
