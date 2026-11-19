"""Tests for energy loss through a layer (isotopia_py/energyloss.py). No cross-section library needed."""

from __future__ import annotations

import io

import numpy as np
import pytest

from isotopia_py import rates, srim
from isotopia_py.energyloss import RangeTable
from isotopia_py.fortran import S
from isotopia_py.state import State

INDEX = srim.table_inputs()
CASES = [(p, z) for p in srim.PARTICLES for z in sorted(INDEX)]


def _range_table(particle, z):
    """RangeTable for a projectile in element z at the data sheet's density and atomic mass."""
    rho, mass = float(INDEX[z]["density_g_cm3"]), float(INDEX[z]["target_mass_u"])
    return RangeTable.from_srim(srim.load_table(particle, z), rho, mass), rho, mass


# ---------------------------------------------------------------------------------------------
# The two directions agree: (E_in, E_out) -> thickness -> E_out, and (E_in, t) -> E_out -> t
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("particle,z", CASES)
def test_exit_energy_inverts_thickness(particle, z):
    r, _, _ = _range_table(particle, z)
    rng = np.random.default_rng(z)
    E_in = np.exp(rng.uniform(np.log(r.emin), np.log(r.emax), 200))
    E_out = np.exp(rng.uniform(np.log(r.emin), np.log(E_in)))
    t = r.thickness(E_in, E_out)
    back = r.exit_energy(E_in, t)
    # R(E_in) - t subtracts nearly equal numbers when E_out is near the floor, which costs a
    # few digits of double precision; 1e-8 is well above that and far below any physics error.
    np.testing.assert_allclose(back, E_out, rtol=1e-8)
    np.testing.assert_allclose(r.thickness(E_in, back), t, rtol=1e-10, atol=1e-30)


@pytest.mark.parametrize("particle,z", [("p", 42), ("a", 42), ("p", 7), ("a", 79)])
def test_thickness_matches_direct_integration(particle, z):
    """Closed-form segments against a plain trapezoid integral of dE/S (manual Eq. 4.23)."""
    r, rho, mass = _range_table(particle, z)
    table = srim.load_table(particle, z)
    for E_in, E_out in ((16.0, 12.0), (5.0, 0.05), (250.0, 1.0)):
        x = np.linspace(np.log(E_out), np.log(E_in), 20001)  # integrate in ln E
        f = np.exp(x) / table.stopping_power(np.exp(x), rho, mass)
        direct = np.sum((f[1:] + f[:-1]) / 2 * np.diff(x))
        assert float(r.thickness(E_in, E_out)) == pytest.approx(direct, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# Agreement with ISOTOPIA: Ebeam and Eback in, thickness out
# ---------------------------------------------------------------------------------------------


def test_reproduces_fortran_thickness_and_inverts_it():
    """p + Mo-100, Ebeam 16, Eback 12, rho 10.22 (sample p-Mo100-Tc099m).

    With ISOTOPIA's own Bethe-Bloch stopping power, thickness(16, 12) must equal the
    "Effective target thickness" in samples/p-Mo100-Tc099m/org/isotopia.out (2.097183E-02 cm),
    and exit_energy(16, that thickness) must return Eback = 12 MeV.
    """
    st = State(out=io.StringIO())
    st.k0, st.Ztarget, st.Atarget, st.rho_target = 2, 42, 100, S(10.22)
    grid = np.geomspace(1.0, 20.0, 401)
    bethe = np.array([float(rates.stoppingpower(st, E)) for E in grid])
    r = RangeTable(grid, bethe)
    fortran_thickness = 2.097183e-02  # cm, 7 printed digits
    # Differences: the Fortran's 100-point midpoint rule and single precision (about 1e-7 each)
    assert float(r.thickness(16.0, 12.0)) == pytest.approx(fortran_thickness, rel=2e-6)
    assert float(r.exit_energy(16.0, fortran_thickness)) == pytest.approx(
        12.0, abs=1e-5
    )


# ---------------------------------------------------------------------------------------------
# Behaviour of a layer
# ---------------------------------------------------------------------------------------------


def test_two_layers_equal_one_layer_of_the_summed_thickness():
    r, _, _ = _range_table("p", 28)
    t1, t2 = 0.01, 0.02
    after_first = r.exit_energy(20.0, t1)
    assert float(r.exit_energy(after_first, t2)) == pytest.approx(
        float(r.exit_energy(20.0, t1 + t2)), rel=1e-10
    )


def test_zero_thickness_and_thicker_layers():
    r, _, _ = _range_table("a", 13)
    assert float(r.exit_energy(30.0, 0.0)) == pytest.approx(30.0, rel=1e-12)
    E_out = r.exit_energy(30.0, np.linspace(0.0, 0.9, 50) * float(r.path_length(30.0)))
    assert np.all(np.diff(E_out) < 0)  # a thicker layer always leaves less energy


def test_particle_that_stops_in_the_layer_has_zero_exit_energy():
    r, _, _ = _range_table("p", 28)
    reach = float(
        r.path_length(10.0)
    )  # path length from 10 MeV down to the table floor
    assert float(r.exit_energy(10.0, reach)) == 0.0
    assert float(r.exit_energy(10.0, 5 * reach)) == 0.0
    just_through = float(r.exit_energy(10.0, reach * (1 - 1e-6)))
    assert r.emin <= just_through < 0.1


def test_arrays_are_handled_elementwise():
    r, _, _ = _range_table("p", 42)
    E_in = np.array([5.0, 16.0, 100.0])
    t = np.array([0.001, 0.02, 0.5])
    out = r.exit_energy(E_in, t)
    assert out.shape == (3,)
    for k in range(3):
        assert out[k] == pytest.approx(float(r.exit_energy(E_in[k], t[k])), rel=1e-12)


def test_invalid_inputs_are_errors():
    r, _, _ = _range_table("p", 28)
    with pytest.raises(ValueError, match="outside the stopping table"):
        r.exit_energy(300.0, 0.1)
    with pytest.raises(ValueError, match="outside the stopping table"):
        r.thickness(10.0, 1e-4)
    with pytest.raises(ValueError, match="E_out must not exceed E_in"):
        r.thickness(10.0, 12.0)
    with pytest.raises(ValueError, match="thickness must be >= 0"):
        r.exit_energy(10.0, -0.1)
    with pytest.raises(ValueError, match="increasing"):
        RangeTable([2.0, 1.0], [10.0, 20.0])
