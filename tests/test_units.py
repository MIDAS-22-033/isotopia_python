"""Unit tests for pieces that can be checked without the cross-section library."""
from __future__ import annotations

import io

import numpy as np
import pytest

from isotopia_py import fortran as F
from isotopia_py.inputdeck import convert, getkeywords
from isotopia_py.kinetics import equations
from isotopia_py.state import State

S = np.float32


def test_es_format_matches_gfortran():
    assert F.es(1.0, 15, 6) == "   1.000000E+00"
    assert F.es(-2.5e-7, 13, 6) == "-2.500000E-07"
    assert F.es(1e100, 15, 6) == "  1.000000+100"
    assert F.es(0.0, 15, 6) == "   0.000000E+00"


def test_f_and_i_formats():
    assert F.f(16.0, 7, 3) == " 16.000"
    assert F.f(0.99999, 8, 5) == " 0.99999"
    assert F.i(7, 3, 3) == "007"
    assert F.i(123456, 3) == "***"


def test_time_breakdown_single_and_double():
    secs = 365.25 * 86400 + 2 * 86400 + 3 * 3600 + 4 * 60 + 5
    parts = F.time_breakdown(secs, F.S(365.25 * 86400), F.S(86400), F.S(3600), F.S(60), single=True)
    assert parts == [1, 2, 3, 4, 5]
    parts = F.time_breakdown(secs, F.S(365.25 * 86400), F.S(86400), F.S(3600), F.S(60), single=False)
    assert parts == [1, 2, 3, 4, 5]


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
    st.Zdepth, st.Adepth = 0, 2
    st.k0 = 2
    st.M_target = S(0.2)
    st.Tir, st.Tco = S(86400.0), S(0.0)
    st.abun[1] = S(1.0)
    st.reaction_rate[(0, 0, -1)] = S(1e-9)
    st.reaction_rate[(43, 99, -1)] = S(2e-10)
    st.lam[(43, 99, -1)] = S(3.2e-5)
    st.Nisomer[(43, 99)] = -1
    st.rpexist[(43, 99, -1)] = True
    st.flagdecay = False
    equations(st)
    N0 = float(st.N_0)
    t = float(st.Tgrid[st.Ntime])
    expected = N0 * 2e-10 * (np.exp(-1e-9 * t) - np.exp(-3.2e-5 * t)) / (3.2e-5 - 1e-9)
    assert float(st.Niso[(43, 99, -1)][st.Ntime]) == pytest.approx(expected, rel=1e-6)
    assert float(st.Niso[(42, 100, -1)][st.Ntime]) == pytest.approx(N0 * np.exp(-1e-9 * t), rel=1e-6)
