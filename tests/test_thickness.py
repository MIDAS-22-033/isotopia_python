"""Tests for 'thickness' as an input and the 'stopping' keyword (charged particles).

These need no cross-section library: they run the input checks and the stopping-power set-up
(run.reacinitial -> rates.setup_stopping) and the beam spectrum, not the reaction rates.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from isotopia_py import rates, srim
from isotopia_py.constants import RHOTABLE, RHOTABLE_FORTRAN
from isotopia_py.energyloss import RangeTable
from isotopia_py.fortran import S
from isotopia_py.inputdeck import checkvalue, read_keywords, readinput
from isotopia_py.run import reacinitial
from isotopia_py.state import IsotopiaError, State

MO100 = ("projectile p", "element Mo", "mass 100", "Ebeam 16.")


def _setup(*lines, fortran_defaults=False):
    """Read and check the input, set the density and stopping power; return the State."""
    st = State(out=io.StringIO(), fortran_defaults=fortran_defaults)
    readinput(st, list(lines))
    read_keywords(st)
    checkvalue(st)
    reacinitial(st)
    return st


def _leff(st):
    """Effective target thickness [cm] from the beam spectrum (manual Eq. 4.23)."""
    rates._charged_particle_spectrum(st)
    return float(st.Leff)


# ---------------------------------------------------------------------------------------------
# The two ways of defining a target agree: (Ebeam, Eback) -> thickness -> Eback
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,rtol",
    [
        ("srim", 1e-6),  # exact path lengths; single precision limits it
        ("bethe", 1e-5),  # the Fortran's 100-point midpoint rule limits it
    ],
)
def test_thickness_input_agrees_with_eback_input(model, rtol):
    by_eback = _setup(*MO100, "Eback 12.", f"stopping {model}")
    thickness = _leff(by_eback)
    by_thickness = _setup(*MO100, f"thickness {thickness!r}", f"stopping {model}")
    assert float(by_thickness.Eback) == pytest.approx(12.0, rel=rtol)
    assert _leff(by_thickness) == pytest.approx(thickness, rel=rtol)


@pytest.mark.parametrize(
    "particle,element,mass", [("p", "Al", 27), ("a", "Au", 197), ("a", "N", 14)]
)
def test_thickness_round_trip_for_other_targets(particle, element, mass):
    lines = (
        f"projectile {particle}",
        f"element {element}",
        f"mass {mass}",
        "Ebeam 30.",
    )
    thickness = _leff(_setup(*lines, "Eback 20."))
    st = _setup(*lines, f"thickness {thickness!r}")
    assert float(st.Eback) == pytest.approx(20.0, rel=1e-6)
    assert _leff(st) == pytest.approx(thickness, rel=1e-6)


def test_srim_target_thickness_matches_energyloss_module():
    """The run and energyloss.RangeTable give the same thickness for the same inputs."""
    st = _setup(*MO100, "Eback 12.")
    # atoms per cm3 use the mass number of the isotope, as the Fortran does
    assert float(st.molar_mass()) == 100.0
    independent = RangeTable.from_srim(
        srim.load_table("p", 42), rho=float(RHOTABLE[42 - 1]), mass=100.0
    )
    assert _leff(st) == pytest.approx(
        float(independent.thickness(16.0, 12.0)), rel=1e-6
    )


# ---------------------------------------------------------------------------------------------
# Which stopping power is used
# ---------------------------------------------------------------------------------------------


def test_default_is_srim_where_a_table_exists():
    st = _setup(*MO100, "Eback 12.")
    assert st.stopping_model == "srim" and st.out.getvalue() == ""
    assert st.rho_target == RHOTABLE[42 - 1]


@pytest.mark.parametrize(
    "lines,what",
    [
        (
            ("projectile h", "element Np", "mass 237", "Ebeam 16.", "Eback 12."),
            "h in Np",
        ),
        (
            ("projectile p", "element Np", "mass 237", "Ebeam 16.", "Eback 12."),
            "p in Np",
        ),
    ],
)
def test_bethe_bloch_with_a_warning_where_no_table_exists(lines, what):
    st = _setup(*lines)
    assert st.stopping_model == "bethe"
    assert f"ISOTOPIA-warning: no SRIM stopping table for {what}" in st.out.getvalue()
    with pytest.raises(IsotopiaError, match="no SRIM stopping table"):
        _setup(*lines, "stopping srim")


def test_stopping_keyword_selects_bethe_bloch():
    st = _setup(*MO100, "Eback 12.", "stopping bethe")
    assert st.stopping_model == "bethe" and st.range_table is None
    with pytest.raises(IsotopiaError, match="stopping must be equal to srim or bethe"):
        _setup(*MO100, "Eback 12.", "stopping ziegler")


def test_fortran_defaults_reproduce_the_fortran_thickness():
    """Sample p-Mo100-Tc099m: 'Effective target thickness [cm]: 2.097183E-02' in its reference."""
    st = _setup(*MO100, "Eback 12.", fortran_defaults=True)
    assert st.stopping_model == "bethe" and st.rho_target == RHOTABLE_FORTRAN[42 - 1]
    assert f"{_leff(st):.6E}" == "2.097183E-02"
    assert st.out.getvalue() == ""


# ---------------------------------------------------------------------------------------------
# Input checks and the beam stopping in the target
# ---------------------------------------------------------------------------------------------


def test_eback_and_thickness_together_are_an_error():
    with pytest.raises(IsotopiaError, match="either Eback or thickness"):
        _setup(*MO100, "Eback 12.", "thickness 0.02")


def test_beam_that_stops_in_the_target():
    st = _setup(*MO100, "thickness 0.1")
    table = st.range_table
    reach = float(table.path_length(16.0))
    assert reach < 0.1
    # Eback is the lowest table energy, and only the irradiated depth is the effective target
    assert float(st.Eback) == pytest.approx(table.emin, rel=1e-6)
    assert _leff(st) == pytest.approx(reach, rel=1e-5)
    text = st.out.getvalue()
    assert "ISOTOPIA-warning: the beam stops in the target" in text
    assert f"{reach:.6E} cm into its 1.000000E-01 cm" in text


def test_bethe_bloch_refuses_a_target_that_stops_the_beam():
    with pytest.raises(IsotopiaError, match="Bethe-Bloch formula is not valid"):
        _setup(*MO100, "thickness 0.1", "stopping bethe")


def test_thickness_for_neutrons_is_unchanged():
    st = _setup("projectile n", "element Mo", "mass 98", "thickness 0.1")
    assert st.thickness == S(0.1) and st.Eback == S(-1)


def test_thinner_target_leaves_more_energy():
    ebacks = [
        float(_setup(*MO100, f"thickness {t}").Eback) for t in (0.005, 0.01, 0.02, 0.04)
    ]
    assert np.all(np.diff(ebacks) < 0) and ebacks[0] < 16.0
