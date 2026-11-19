"""Tests for the SRIM stopping-table reader (isotopia_py/srim.py). No cross-section library needed."""

from __future__ import annotations

import re

import numpy as np
import pytest

from isotopia_py import srim
from isotopia_py.constants import PARMASS, PARSYM, RHOTABLE

INDEX = srim.table_inputs()
CASES = [(p, z) for p in srim.PARTICLES for z in sorted(INDEX)]


def _sheet(z):
    """Density (g/cm3) and atomic mass (u) the table for element z was made with."""
    return float(INDEX[z]["density_g_cm3"]), float(INDEX[z]["target_mass_u"])


def test_index_covers_the_natural_elements():
    # every element SRIM can take as a target: H to U (Z = 1-92)
    natural = list(range(1, 93))
    assert sorted(INDEX) == natural


@pytest.mark.parametrize("particle,z", CASES)
def test_every_table_loads_and_matches_its_inputs(particle, z):
    t = srim.load_table(particle, z)
    rho, mass = _sheet(z)
    assert (t.Z, t.symbol) == (z, INDEX[z]["symbol"])
    assert t.gas == (INDEX[z]["srim_phase"] == "gas")
    assert (
        len(t.energy) == 142
        and t.emin == pytest.approx(1e-3, rel=1e-5)
        and t.emax == 250.0
    )
    # header density and atoms/cm3 are the sheet's density and N_A*rho/M (5 printed digits)
    assert t.srim_density == pytest.approx(rho, rel=5e-5)
    assert t.srim_atom_density == pytest.approx(srim.N_A * rho / mass, rel=1e-4)


@pytest.mark.parametrize("particle,z", CASES)
def test_conversion_matches_srims_own_factor(particle, z):
    """Our eV/(1E15 atoms/cm2) -> MeV/cm conversion against the MeV/mm factor SRIM prints."""
    t = srim.load_table(particle, z)
    factor = float(
        re.search(
            r"^\s*(\S+)\s+MeV\s*/\s*mm",
            t.path.read_text(encoding="latin-1"),
            re.MULTILINE,
        )[1]
    )
    rho, mass = _sheet(z)
    ours = t.stopping_power(t.energy, rho, mass)  # MeV/cm
    srims = (t.electronic + t.nuclear) * factor * 10.0  # MeV/mm -> MeV/cm
    np.testing.assert_allclose(ours, srims, rtol=1e-4)  # factor has 5 digits


def test_interpolation_returns_table_values_at_table_energies():
    t = srim.load_table("p", 28)
    np.testing.assert_allclose(
        t.per_atom(t.energy, "electronic"), t.electronic, rtol=1e-12
    )
    np.testing.assert_allclose(t.per_atom(t.energy, "nuclear"), t.nuclear, rtol=1e-12)
    np.testing.assert_allclose(
        t.per_atom(t.energy), t.electronic + t.nuclear, rtol=1e-12
    )


def test_interpolation_lies_between_neighbouring_rows():
    t = srim.load_table("a", 79)
    mid = np.sqrt(t.energy[:-1] * t.energy[1:])  # geometric midpoints
    s, tot = t.per_atom(mid), t.electronic + t.nuclear
    assert np.all(s >= np.minimum(tot[:-1], tot[1:]) * (1 - 1e-12))
    assert np.all(s <= np.maximum(tot[:-1], tot[1:]) * (1 + 1e-12))


def test_stopping_power_scales_with_atoms_per_cm3():
    """Enriched target, same lattice: rho and M scale together, so atoms/cm3 and S are unchanged."""
    t = srim.load_table("p", 42)
    rho, mass = _sheet(42)
    s_nat = t.stopping_power(16.0, rho, mass)
    assert t.stopping_power(16.0, rho * 99.9 / mass, 99.9) == pytest.approx(
        s_nat, rel=1e-12
    )
    assert t.stopping_power(16.0, 2 * rho, mass) == pytest.approx(2 * s_nat, rel=1e-12)


@pytest.mark.parametrize("E", [5e-4, 251.0])
def test_energy_outside_table_is_an_error(E):
    with pytest.raises(ValueError, match="outside the SRIM table"):
        srim.load_table("p", 28).per_atom(E)


def test_unavailable_tables_are_errors():
    with pytest.raises(FileNotFoundError, match="no SRIM table for p in Np"):
        srim.load_table("p", 93)
    with pytest.raises(ValueError, match="no SRIM tables for projectile 'n'"):
        srim.load_table("n", 28)


@pytest.mark.parametrize("heavy,light", [("d", "p"), ("t", "p"), ("h", "a")])
def test_electronic_stopping_scales_with_velocity(heavy, light):
    """Ions of the same charge at the same velocity have the same electronic stopping, so
    S_e of d or t at E * m/m_p equals S_e of p at E (He-3 vs alpha alike). The tables are
    separate SRIM runs, so this checks them against each other; the limit covers the four
    printed digits and the interpolation (worst seen: 0.38%, He-3 vs alpha)."""
    k = {p: i for i, p in enumerate(PARSYM)}
    ratio = PARMASS[k[heavy]] / PARMASS[k[light]]
    E = np.geomspace(0.01, 80.0, 25)
    for z in INDEX:
        slow, fast = srim.load_table(light, z), srim.load_table(heavy, z)
        assert np.allclose(
            fast.per_atom(E * ratio, "electronic"),
            slow.per_atom(E, "electronic"),
            rtol=5e-3,
        ), z


def test_density_table_matches_srim_inputs():
    """constants.RHOTABLE and the SRIM tables come from the same element data sheet."""
    for z in INDEX:
        assert RHOTABLE[z - 1] == np.float32(_sheet(z)[0]), INDEX[z]["symbol"]


# A table as written by SRIM's stopping-table window (header style differs from the SR Module).
_WINDOW_TABLE = """\
 Ion = Hydrogen [1] , Mass = 1.008 amu

 Target Density =  8.8955E+00 g/cm3 = 9.1274E+22 atoms/cm3
 ======= Target  Composition ========
    Atom   Atom   Atomic    Mass
    Name   Numb   Percent   Percent
    ----   ----   -------   -------
     Ni     28    100.00    100.00
 ====================================
 Bragg Correction = 0.00%
 Stopping Units =   eV / (1E15 atoms/cm2)
  10.00 keV   1.059E+01  2.588E-01     609 A       421 A       351 A
  11.00 keV   1.105E+01  2.476E-01     668 A       443 A       373 A
"""


def test_reads_srim_window_format_and_checks_the_ion(tmp_path):
    f = tmp_path / "h_in_ni.txt"
    f.write_text(_WINDOW_TABLE.replace("Mass = 1.008", "Mass = 1.007825"))
    t = srim.read_table(f, "p")
    assert (t.Z, t.symbol, t.gas) == (28, "Ni", False)
    np.testing.assert_allclose(t.energy, [0.010, 0.011])
    np.testing.assert_allclose(t.electronic, [10.59, 11.05])
    f.write_text(_WINDOW_TABLE)  # SRIM's default H mass, not ISOTOPIA's p
    with pytest.raises(ValueError, match="ion in the header"):
        srim.read_table(f, "p")


def test_rejects_other_stopping_units(tmp_path):
    f = tmp_path / "mass_units.txt"
    f.write_text(
        _WINDOW_TABLE.replace("Mass = 1.008", "Mass = 1.007825").replace(
            "eV / (1E15 atoms/cm2)", "MeV / (mg/cm2)"
        )
    )
    with pytest.raises(ValueError, match="must be in eV/\\(1E15 atoms/cm2\\)"):
        srim.read_table(f, "p")
