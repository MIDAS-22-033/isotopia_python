"""Tests for reactions on products (daughters.py) and the solver for networks with cycles.

1. Made-up data (no library): the rate sum, the window, a second hit against
   its closed-form solution, the 'outside_window' bin, and a stiff cycle
   (quasi-steady state) against its closed form.
2. Sample cases (need isotopia/ and isotopia.libs/, else skipped):
   * the rebuilt weights reproduce every rate the port computed, all samples;
   * p + Mo-100 with daughter hits (needs the daughters' library folders):
     second-generation products agree with first-order perturbation theory,
     computed from the run without hits; warnings are present.
"""
from __future__ import annotations

import numpy as np
import pytest
from fakes import LIBS, SAMPLES, FakeLibrary, run_sample_capturing, sample_cases

from isotopia_py.daughters import (
    RateGrid,
    Window,
    _keep_inside,
    burnup_file,
    check_rate_grid,
    hit_reactions,
    rate_grid,
)
from isotopia_py.decaylib import OUTSIDE, DecayLibrary, label
from isotopia_py.irradiation import Irradiation, irradiate, isotopia_times
from isotopia_py.matrix import Reactions, build_chain

# ----------------------------------------------------------------------------
# 1. Made-up data
# ----------------------------------------------------------------------------

T, D, E, F, X, Y = (10, 20, 0), (11, 20, 0), (12, 21, 0), (12, 20, 0), (9, 20, 0), (8, 20, 0)
N0, T_IRR, T_COOL = 1e21, 86400.0, 86400.0
TIMES = np.linspace(0.0, T_IRR + T_COOL, 41)


def test_rate_grid_sum_and_range():
    """R = scale * sum w_k sigma(E_k) * 1e-27, linear interpolation, zero outside [E_0, E_last)."""
    grid = RateGrid(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.0, 2.0, 4.0]), 1e10)
    E, xs = [1.5, 2.5, 4.0], [10.0, 30.0, 50.0]          # 4.0 is the end: excluded
    sigma = np.array([0.0, 20.0, 37.5 + 0.0, 0.0])        # at 2 -> 20, at 3 -> 30 + 0.5*... see below
    sigma[2] = 30.0 + (3.0 - 2.5) / (4.0 - 2.5) * 20.0
    assert grid.rate(E, xs) == pytest.approx(1e10 * np.dot(grid.weights, sigma) * 1e-27, rel=1e-14)
    assert grid.rate([5.0, 6.0], [1.0, 1.0]) == 0.0


def test_burnup_file_follows_isotopia():
    """MT001 (total) for neutrons with self-shielding, MT003 without, MT005 otherwise."""
    from types import SimpleNamespace
    st = SimpleNamespace(k0=1, flagselfshield=True, crosspath="/libs/", libname="iaea.2026")
    assert burnup_file(st, (71, 177, 0)) == "/libs/n/Lu177/iaea.2026/tables/xs/n-Lu177-MT001.iaea.2026"
    st.flagselfshield = False
    assert burnup_file(st, (71, 177, 0)).endswith("n-Lu177-MT003.iaea.2026")
    st.k0 = 2
    assert burnup_file(st, (47, 110, 1)) == "/libs/p/Ag110m/iaea.2026/tables/xs/p-Ag110m-MT005.iaea.2026"


def test_window():
    w = Window(A_below=3, A_above=1, Z_below=2, Z_above=0)
    assert w.contains(T, (10, 17, 0)) and w.contains(T, (8, 21, 0))
    assert not w.contains(T, (10, 16, 0)) and not w.contains(T, (11, 20, 0))
    with pytest.raises(ValueError):
        Window(-1, 0, 0, 0)


def test_second_hit_closed_form():
    """T -> D (reaction), D -> E (reaction, beam on only), D -> F (decay); E and F stable.

    k = lambda_D + R_D during irradiation:
      N_D = N0 R_TD (e^-R_T t - e^-k t) / (k - R_T)
      N_E = R_DE * integral of N_D
    """
    R_T, R_TD, R_D, R_DE, lam_D = 1e-4, 1e-4, 4e-5, 4e-5, 2e-5
    lib = FakeLibrary({T: (0.0, []), D: (lam_D, [(F, 1.0)]), E: (0.0, []), F: (0.0, [])})
    reactions = {T: Reactions(R_T, {D: R_TD}), D: Reactions(R_D, {E: R_DE})}
    irr = Irradiation(T, N0, R_T, {D: R_TD}, T_IRR, T_COOL)
    inv = irradiate(irr, TIMES, lib, reactions=reactions)

    k = lam_D + R_D
    on = np.minimum(TIMES, T_IRR)
    nD = N0 * R_TD * (np.exp(-R_T * on) - np.exp(-k * on)) / (k - R_T)
    integral = N0 * R_TD / (k - R_T) * (-np.expm1(-R_T * on) / R_T + np.expm1(-k * on) / k)
    tau = np.maximum(TIMES - T_IRR, 0.0)
    np.testing.assert_allclose(inv[D], nD * np.exp(-lam_D * tau), rtol=1e-11)
    np.testing.assert_allclose(inv[E], R_DE * integral, rtol=1e-11)     # no hits in cooling
    np.testing.assert_allclose(inv.atoms.sum(axis=1), N0, rtol=1e-13)   # rates balance


def test_outside_window_bin_keeps_atoms():
    lost = {}
    rx = _keep_inside(Reactions(1e-4, {D: 6e-5, X: 4e-5}), T, Window(0, 0, 0, 1), lost)
    assert rx.products == {D: 6e-5, OUTSIDE: 4e-5} and lost == {X: 4e-5}
    lib = FakeLibrary({T: (0.0, []), D: (0.0, [])})
    inv = irradiate(Irradiation(T, N0, 1e-4, rx.products, T_IRR, T_COOL), TIMES, lib)
    np.testing.assert_allclose(inv.atoms.sum(axis=1), N0, rtol=1e-13)
    assert label(OUTSIDE) == "outside_window" and inv[OUTSIDE][-1] > 0


def test_stiff_cycle_quasi_steady_state():
    """T -> X (reaction), X -> Y (decay), Y -> T (decay, 1e-7 s): a stiff cycle.

    With Y instantaneous, N_T = N0 (lam + R e^-(R + lam) t) / (R + lam), lam = lambda_X;
    the error of that limit is of order lambda_X / lambda_Y = 3e-12.
    """
    R, lam_X, lam_Y = 1e-4, 3e-5, 1e7
    lib = FakeLibrary({T: (0.0, []), X: (lam_X, [(Y, 1.0)]), Y: (lam_Y, [(T, 1.0)])})
    inv = irradiate(Irradiation(T, N0, R, {X: R}, T_IRR, T_COOL), TIMES, lib)
    on, tau = np.minimum(TIMES, T_IRR), np.maximum(TIMES - T_IRR, 0.0)
    nT_irr = N0 * (lam_X + R * np.exp(-(R + lam_X) * on)) / (R + lam_X)
    nX_eoi = N0 - N0 * (lam_X + R * np.exp(-(R + lam_X) * T_IRR)) / (R + lam_X)
    expected_T = np.where(TIMES <= T_IRR, nT_irr, N0 - nX_eoi * np.exp(-lam_X * tau))
    np.testing.assert_allclose(inv[T], expected_T, rtol=1e-10)
    np.testing.assert_allclose(inv[Y], lam_X * inv[X] / lam_Y, rtol=1e-9)
    assert any("quasi-steady state" in w for w in inv.warnings)


# ----------------------------------------------------------------------------
# 2. Sample cases
# ----------------------------------------------------------------------------

needs_samples = pytest.mark.skipif(not (LIBS.is_dir() and SAMPLES.is_dir()),
                                   reason="needs isotopia/ and isotopia.libs/")
needs_daughters = pytest.mark.skipif(
    not all((LIBS / "p" / n / "iaea.2026").is_dir() for n in ("Mo099", "Tc099", "Mo098", "Mo097")),
    reason="needs the library folders of Mo-100's daughters (p/Mo099, p/Tc099, ...)")


@needs_samples
@pytest.mark.parametrize("case", sample_cases())
def test_rate_grid_reproduces_port_rates(case, tmp_path, monkeypatch):
    """Every rate of every isotope run, R_T included, to 1e-5 (raises RateGridError if not)."""
    checked = []
    run_sample_capturing(case, tmp_path, monkeypatch,
                         lambda st: (check_rate_grid(st, rate_grid(st)), checked.append(st.iso)))
    assert checked


@needs_samples
@needs_daughters
def test_mo100_second_generation_matches_perturbation_theory(tmp_path, monkeypatch):
    """Products made only by second hits, against first-order perturbation theory.

    For a stable product P that the target cannot make and that nothing
    decays into, at end of irradiation
        N_P = sum_D R_D->P * integral_0^t_irr N_D(t) dt
    with N_D from the run without daughter hits. Corrections are of order
    R * t_irr (8.6e-5 here), so agreement to 1e-3 is a meaningful test.
    Products fed by decay are left out: e.g. Tc-97 also gets 8e-4 of its atoms
    from Tc-97m, itself made by second hits, which this formula ignores.
    """
    found = []
    run_sample_capturing("p-Mo100-Tc099m", tmp_path, monkeypatch, found.append)
    st = found[0]
    library = DecayLibrary()
    irr = Irradiation.from_state(st)
    reactions, warnings = hit_reactions(st, irr, Window(8, 2, 3, 2), library)
    hits = irradiate(irr, [irr.t_irr], library, reactions=reactions)
    net = build_chain([irr.target], library, reactions=reactions)
    fed_by_decay = {n for i, n in enumerate(net.nuclides)
                    if np.any(np.delete(net.A[i], i))}
    fine = np.linspace(0.0, irr.t_irr, 2001)
    base = irradiate(irr, fine, library)

    # Only inventories above 1e-12 of the target: below that the solver is at its
    # rounding floor (measured against a 50-digit reference: 2e-11 above 1e-12).
    second = [n for n in hits.nuclides if n != OUTSIDE and n not in base.nuclides
              and hits.lam[hits.nuclides.index(n)] < 1e-12 and hits[n][0] > 1e-12 * irr.N0
              and n not in fed_by_decay]
    assert second, "expected stable products made only by second hits"
    for p in second:
        expected = sum(rx.products.get(p, 0.0) * np.trapezoid(base[d], fine)
                       for d, rx in reactions.items() if d != irr.target and d in base.nuclides)
        assert hits[p][0] == pytest.approx(expected, rel=1e-3), label(p)

    text = " ".join(warnings)
    assert "not hit" in text and "outside_window" in text and "own" in text


@needs_samples
@needs_daughters
def test_mo100_target_and_window_bin(tmp_path, monkeypatch):
    """The target follows its own rates; atoms leaving the window are counted.

    Nothing feeds Mo-100, so N_T = N0 exp(-R_T t) exactly. With hits the network
    has cycles and takes the general expm path, measured at 2e-11 against a
    50-digit reference (matrix._propagate_cyclic); 1e-10 allows for that.
    """
    found = []
    run_sample_capturing("p-Mo100-Tc099m", tmp_path, monkeypatch, found.append)
    st = found[0]
    library = DecayLibrary()
    irr = Irradiation.from_state(st)
    times = isotopia_times(st)
    reactions, _ = hit_reactions(st, irr, Window(8, 2, 3, 2), library)
    hits = irradiate(irr, times, library, reactions=reactions)
    base = irradiate(irr, times, library)
    exact = irr.N0 * np.exp(-irr.burnup * np.minimum(times, irr.t_irr))
    np.testing.assert_allclose(base[irr.target], exact, rtol=1e-13)
    np.testing.assert_allclose(hits[irr.target], exact, rtol=1e-10)
    assert hits[OUTSIDE][st.Ntime] > 0
    assert reactions[irr.target].loss == irr.burnup


@needs_samples
@pytest.mark.skipif(not (LIBS / "n" / "Lu177" / "iaea.2026").is_dir(),
                    reason="needs the library folder n/Lu177")
def test_neutron_daughter_burnup_is_total_cross_section(tmp_path, monkeypatch):
    """A hit neutron daughter depletes at its total cross section, like ISOTOPIA's target."""
    from isotopia_py.data import _read_xs_table
    found = []
    run_sample_capturing("n-Lu176-Lu177", tmp_path, monkeypatch, found.append)
    st = found[0]
    irr = Irradiation.from_state(st)
    reactions, warnings = hit_reactions(st, irr, Window(2, 2, 1, 1))
    lu177 = (71, 177, 0)
    total = rate_grid(st).rate(*_read_xs_table(burnup_file(st, lu177)))
    assert burnup_file(st, lu177).endswith("-MT001.iaea.2026")
    assert reactions[lu177].loss == pytest.approx(total, rel=1e-12)
    assert reactions[lu177].loss >= sum(reactions[lu177].products.values())
    # The library's Yb-177 ground and isomer files add up to more than its total file.
    assert any("Isomer-state files" in w and "Yb176 -> Yb177" in w for w in warnings)
