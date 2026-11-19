"""Tests for target reactions in the matrix solver (irradiation.py, isotopia_compat.py).

1. Full mode on made-up data (no library needed), against closed-form
   solutions of manual Eq. 4.1 with reactions on the target only.
2. Compatibility mode against the port's kinetics.equations on a made-up
   State: double-counted feeding, cooling feeding order and isomers.
3. Sample cases (need isotopia/ and isotopia.libs/, else skipped):
   * compatibility mode against the port, every nuclide and time, all samples;
   * full mode against the port for products that neither code feeds by decay;
   * warnings and missing-data handling.
"""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest
from fakes import (
    DECAY_DIR,
    LIBS,
    SAMPLES,
    FakeLibrary,
    run_sample_capturing,
    sample_cases,
)

from isotopia_py import kinetics
from isotopia_py.decaylib import DecayDataError, DecayLibrary, label
from isotopia_py.fortran import S
from isotopia_py.irradiation import Irradiation, irradiate, isotopia_times, product_keys
from isotopia_py.isotopia_compat import _beta_parents, compat_inventory, port_feeders
from isotopia_py.matrix import Reactions, build_chain, propagate
from isotopia_py.state import State

# The port prints and stores float32 (relative precision 6e-8); 1e-5 is the
# same tolerance the regression tests use for reference outputs.
RTOL = 1e-5
# float32 values close to underflow (1.2e-38) lose digits: compare above this.
FLOOR = 1e-30

# ----------------------------------------------------------------------------
# 1. Full mode, made-up data
# ----------------------------------------------------------------------------

T, X, Y, Z, W = (10, 20, 0), (11, 20, 0), (12, 20, 0), (9, 20, 0), (9, 19, 0)
R_T, R_X, R_Y, R_W = 1e-4, 4e-5, 3e-5, 3e-5    # s^-1; products sum to R_T
LAM_X, BR_Y = 2e-5, 0.7                           # X decays to Y (0.7) and Z (0.3)
N0, T_IRR, T_COOL = 1e21, 86400.0, 86400.0
TIMES = np.linspace(0.0, T_IRR + T_COOL, 41)      # includes t_irr itself


def network_lib(lam_target=0.0):
    """Target T (stable unless lam_target > 0, then decays to W) and its products."""
    return FakeLibrary({
        T: (lam_target, [(W, 1.0)] if lam_target else []),
        X: (LAM_X, [(Y, BR_Y), (Z, 1 - BR_Y)]),
        Y: (0.0, []), Z: (0.0, []), W: (0.0, []),
    })


def make_irr(rates=None, burnup=R_T):
    return Irradiation(T, N0, burnup, rates or {X: R_X, Y: R_Y, W: R_W}, T_IRR, T_COOL)


def closed_form(t):
    """Manual Eq. 4.1 solved by hand for this network: irradiation, then cooling."""
    def irr(t):
        f_T = -np.expm1(-R_T * t) / R_T              # integral of exp(-R_T t')
        f_X = -np.expm1(-LAM_X * t) / LAM_X
        nX = N0 * R_X * (np.exp(-R_T * t) - np.exp(-LAM_X * t)) / (LAM_X - R_T)
        fed = LAM_X * N0 * R_X / (LAM_X - R_T) * (f_T - f_X)   # atoms of X that decayed
        return {T: N0 * np.exp(-R_T * t), X: nX, W: N0 * R_W * f_T,
                Y: N0 * R_Y * f_T + BR_Y * fed, Z: (1 - BR_Y) * fed}

    on = np.minimum(t, T_IRR)
    tau = np.maximum(t - T_IRR, 0.0)
    eoi, now = irr(T_IRR), irr(on)
    decayed = eoi[X] * -np.expm1(-LAM_X * tau)
    cool = {T: eoi[T], W: eoi[W], X: eoi[X] * np.exp(-LAM_X * tau),
            Y: eoi[Y] + BR_Y * decayed, Z: eoi[Z] + (1 - BR_Y) * decayed}
    return {k: np.where(t <= T_IRR, now[k], cool[k]) for k in now}


def test_irradiation_and_cooling_closed_form():
    inv = irradiate(make_irr(), TIMES, network_lib())
    expected = closed_form(TIMES)
    for n in (T, X, Y, Z, W):
        np.testing.assert_allclose(inv[n], expected[n], rtol=1e-11, atol=1e-6, err_msg=label(n))


def test_atoms_conserved_when_rates_balance():
    inv = irradiate(make_irr(), TIMES, network_lib())
    np.testing.assert_allclose(inv.atoms.sum(axis=1), N0, rtol=1e-13)
    assert not inv.warnings


def test_unbalanced_rates_lose_atoms_as_documented():
    """Products receiving only half of R_T: the other half of burnup leaves the bookkeeping."""
    inv = irradiate(make_irr({X: R_X / 2, Y: R_Y / 2, W: R_W / 2}), TIMES, network_lib())
    burned = -np.expm1(-R_T * np.minimum(TIMES, T_IRR))
    np.testing.assert_allclose(inv.atoms.sum(axis=1), N0 * (1 - 0.5 * burned), rtol=1e-13)


def test_unstable_target_decays_in_both_phases():
    lam_T = 3e-6
    inv = irradiate(make_irr(), TIMES, network_lib(lam_T))
    on, tau = np.minimum(TIMES, T_IRR), np.maximum(TIMES - T_IRR, 0.0)
    np.testing.assert_allclose(inv[T], N0 * np.exp(-(R_T + lam_T) * on - lam_T * tau), rtol=1e-12)


def test_decay_back_into_target():
    """T -> X by reaction, X -> T by decay: dN_T/dt = -R N_T + lambda N_X.

    Closed form: N_T = N0 (lambda + R exp(-(R + lambda) t)) / (R + lambda).
    """
    R, lam = 1e-4, 3e-5
    lib = FakeLibrary({T: (0.0, []), X: (lam, [(T, 1.0)])})
    inv = irradiate(Irradiation(T, N0, R, {X: R}, T_IRR, T_COOL), TIMES, lib)
    on, tau = np.minimum(TIMES, T_IRR), np.maximum(TIMES - T_IRR, 0.0)
    nT_irr = N0 * (lam + R * np.exp(-(R + lam) * on)) / (R + lam)
    nX_eoi = N0 - N0 * (lam + R * np.exp(-(R + lam) * T_IRR)) / (R + lam)
    expected = np.where(TIMES <= T_IRR, nT_irr, N0 - nX_eoi * np.exp(-lam * tau))
    np.testing.assert_allclose(inv[T], expected, rtol=1e-12)
    assert any("cycle" in w for w in inv.warnings)


def test_no_burnup_is_pure_decay():
    """With all rates zero the target is untouched and nothing is made."""
    inv = irradiate(make_irr({X: 0.0}, burnup=0.0), TIMES, network_lib())
    np.testing.assert_allclose(inv[T], N0, rtol=1e-15)
    assert np.all(inv[X] == 0.0)


def test_from_state_maps_states_and_warns():
    """Isomer states, the total-only ground state, the target's own rate, and the warnings."""
    st = State()
    st.Ztarget, st.Atarget, st.Zcomp, st.Acomp, st.Zdepth, st.Adepth = 42, 100, 43, 101, 2, 2
    st.N_0, st.Tir, st.Tco = S(1e21), S(3600), S(7200)
    st.Nisomer[(43, 100)] = 1
    rates = {(0, 0, -1): 4.5e-10, (43, 101, -1): 1e-10, (43, 100, -1): 2e-10,
             (43, 100, 0): 1.5e-10, (43, 100, 1): 0.6e-10, (42, 100, -1): 0.5e-10}
    for k, r in rates.items():
        st.reaction_rate[k] = S(r)
        st.rpexist[k] = True
    irr = Irradiation.from_state(st)
    assert irr.target == (42, 100, 0) and irr.N0 == pytest.approx(1e21, rel=1e-7)
    assert irr.t_irr == 3600.0 and irr.t_cool == 7200.0
    assert irr.rates == pytest.approx({(43, 101, 0): 1e-10, (43, 100, 0): 1.5e-10,
                                       (43, 100, 1): 0.6e-10}, rel=1e-7)
    text = " ".join(irr.warnings)
    assert "Tc100" in text and "state rates are used" in text
    assert "leaving Mo100 unchanged" in text
    assert "sum to 0.6889 of R_T" in text


def test_from_state_with_zero_burnup():
    """R_T = 0 (e.g. beam below every threshold) must not divide by zero."""
    st = State()
    st.Ztarget, st.Atarget, st.Zcomp, st.Acomp, st.Zdepth, st.Adepth = 42, 100, 43, 101, 2, 2
    st.reaction_rate[(42, 100, -1)], st.rpexist[(42, 100, -1)] = S(1e-12), True
    irr = Irradiation.from_state(st)
    assert irr.burnup == 0.0 and irr.rates == {} and irr.warnings == ()


# ----------------------------------------------------------------------------
# 2. Compatibility mode against the port, made-up State
# ----------------------------------------------------------------------------

def synthetic_state(flagdecay=True):
    """A State with the feeding cases of equations.f90, ready for kinetics.equations.

    Target Ne-20-like (Z=10, A=20); products:
    P (11,19) beta-plus parent of D, computed before D -> feeds in both phases;
    Q (9,19) beta-minus parent of D with total and ground both feeding (defect 2),
      computed after D -> no feeding in cooling (defect 4), plus an isomer that
      decays by IT (rtyp 3), which feeds nothing (defect 3);
    E (10,18) with lambda one float32 step from R_T (near-singular closed form);
    (12,21) in the first row without a cross section: present, stays zero.
    """
    st = State()
    st.Ztarget, st.Atarget, st.Zcomp, st.Acomp, st.Zdepth, st.Adepth = 10, 20, 11, 21, 3, 3
    st.M_target, st.Tir, st.Tco, st.flagdecay = S(1e-3), S(86400), S(172800), flagdecay
    R_T = S(2e-5)
    st.reaction_rate[(0, 0, -1)] = R_T
    st.Nisomer[(9, 19)] = 1
    data = {   # key: (R_T->i, lambda, rtyp)
        (11, 19, -1): (2e-6, 3e-5, 2), (10, 19, -1): (1e-6, 1e-5, 0),
        (9, 19, -1): (3e-6, 5e-5, 1), (9, 19, 0): (2e-6, 5e-5, 1), (9, 19, 1): (1e-6, 1e-3, 3),
        (10, 18, -1): (1e-6, float(np.nextafter(R_T, np.float32(1))), 0),
        (10, 20, -1): (5e-7, 0.0, 0),
    }
    for k, (r, lam, rtyp) in data.items():
        st.reaction_rate[k], st.lam[k], st.rtyp[k], st.rpexist[k] = S(r), S(lam), rtyp, True
    return st


@pytest.mark.parametrize("flagdecay", [True, False])
def test_compat_matches_port_on_synthetic_state(flagdecay):
    st = synthetic_state(flagdecay)
    kinetics.equations(st)
    compat = compat_inventory(st)
    assert set(compat) == set(product_keys(st))
    for k, c in compat.items():
        np.testing.assert_allclose(c, st.Niso[k], rtol=RTOL, atol=FLOOR, err_msg=str(k))
    assert np.all(compat[(12, 21, -1)] == 0)


def test_port_feeders_reproduce_defects():
    st = synthetic_state()
    feed_irr, feed_cool = port_feeders(st)
    D = (10, 19, -1)
    assert sorted(feed_irr[D]) == [(9, 19, -1), (9, 19, 0), (11, 19, -1)]   # total + ground
    assert feed_cool[D] == [(11, 19, -1)]                                     # no beta-minus


# ----------------------------------------------------------------------------
# 3. Sample cases
# ----------------------------------------------------------------------------

needs_samples = pytest.mark.skipif(
    not (LIBS.is_dir() and SAMPLES.is_dir() and DECAY_DIR.is_dir()),
    reason="needs isotopia/ (samples, decay data) and isotopia.libs/")


def port_exact(st, key, it) -> float:
    """The port's closed-form equations (equations.f90) for ``key`` at grid point ``it``,
    evaluated with 40 significant digits from the same float32 inputs.

    Used only to decide who is right where the port and compatibility mode
    disagree: the port's double-precision closed form loses digits when it
    subtracts nearly equal exponentials (defect 6).
    """
    getcontext().prec = 40

    def dec(x):
        return Decimal(float(x))

    def chain(constants, t):
        """sum_j exp(-c_j t) / prod_(k != j) (c_k - c_j): Bateman sum; ties nudged by 1e-30."""
        cs = list(constants)
        for a in range(len(cs)):
            for b in range(a):
                if cs[a] == cs[b]:
                    cs[a] = cs[a] * (1 + Decimal("1e-30")) + Decimal("1e-60")
        total = Decimal(0)
        for j, cj in enumerate(cs):
            den = Decimal(1)
            for k, ck in enumerate(cs):
                if k != j:
                    den *= ck - cj
            total += (-cj * t).exp() / den
        return total

    keys = product_keys(st)
    position = {k: n for n, k in enumerate(keys)}
    target, R_T, N_0 = (st.Ztarget, st.Atarget, -1), dec(st.reaction_rate[(0, 0, -1)]), dec(st.N_0)

    def irradiated(k, t):
        if k == target:
            return N_0 * (-R_T * t).exp()
        lam_i = dec(st.lam[k])
        value = N_0 * dec(st.reaction_rate[k]) * chain([R_T, lam_i], t)
        for p in (_beta_parents(st, k) if st.flagdecay else ()):
            R_p, lam_p = dec(st.reaction_rate[p]), dec(st.lam[p])
            if R_p > 0 and lam_p > 0:
                value += N_0 * R_p * lam_p * chain([R_T, lam_p, lam_i], t)
        return value

    if it <= st.Ntime:
        return float(irradiated(key, dec(st.Tgrid[it])))
    t_irr, tau = dec(st.Tgrid[st.Ntime]), dec(st.Tgrid[it]) - dec(st.Tir)
    if key == target:
        return float(irradiated(key, t_irr))
    lam_i = dec(st.lam[key])
    value = irradiated(key, t_irr) * (-lam_i * tau).exp()
    for p in (_beta_parents(st, key) if st.flagdecay else ()):
        if p in position and position[p] < position[key]:
            lam_p = dec(st.lam[p])
            value += irradiated(p, t_irr) * lam_p * chain([lam_p, lam_i], tau)
    return float(value)


@needs_samples
@pytest.mark.parametrize("case", sample_cases())
def test_compat_matches_port_on_samples(case, tmp_path, monkeypatch):
    """Every nuclide at every time, for every isotope run of the sample.

    Where port and compatibility mode differ by more than RTOL, the 40-digit
    value of the port's own equations decides: compatibility mode must match
    it to RTOL, and the port must be the one that is off.
    """
    unexplained, port_off, checked = [], [], 0

    def capture(st):
        nonlocal checked
        for k, c in compat_inventory(st).items():
            port = np.asarray(st.Niso[k], dtype=float)
            for it in np.nonzero((np.abs(port) > FLOOR) | (np.abs(c) > FLOOR))[0]:
                checked += 1
                if abs(c[it] - port[it]) <= RTOL * abs(port[it]):
                    continue
                exact = port_exact(st, k, it)
                entry = (case, st.iso, k, int(it), port[it], c[it], exact)
                if abs(c[it] - exact) <= RTOL * abs(exact) < abs(port[it] - exact):
                    port_off.append(entry)
                else:
                    unexplained.append(entry)

    run_sample_capturing(case, tmp_path, monkeypatch, capture)
    assert checked > 0
    assert not unexplained, unexplained[:5]


def _fed_nuclides(net, target):
    """Nuclides that receive atoms by decay from something that has atoms."""
    links = (net.A + net.R != 0) & ~np.eye(len(net.nuclides), dtype=bool)   # [i, j]: j feeds i
    reached = np.zeros(len(net.nuclides), dtype=bool)
    reached[net.index[target]] = True
    while True:   # everything the target can turn into
        more = links[:, reached].any(axis=1) | reached
        if (more == reached).all():
            break
        reached = more
    decay_links = (net.A != 0) & ~np.eye(len(net.nuclides), dtype=bool)
    return {n for n, f in zip(net.nuclides, decay_links[:, reached].any(axis=1)) if f}


def _irradiate_without_target_feeding(irr, times, library):
    """Full mode with the decay links back into the target removed.

    ISOTOPIA never feeds the target (e.g. I-124 -> Te-124 when irradiating
    Te-124). Removing only those links isolates the reaction terms, which must
    then reproduce the port for every product that neither code feeds by decay.
    Target refeeding itself is checked in test_decay_back_into_target.
    """
    net = build_chain([irr.target, *irr.rates], library, "warn",
                      reactions={irr.target: Reactions(irr.burnup, irr.rates)})
    t = net.index[irr.target]
    diagonal = net.A[t, t]
    net.A[t, :] = 0.0
    net.A[t, t] = diagonal
    N0 = np.zeros(len(net.nuclides))
    N0[t] = irr.N0
    on = times <= irr.t_irr
    atoms = np.empty((len(times), len(N0)))
    atoms[on] = propagate(net.A + net.R, N0, times[on])
    eoi = propagate(net.A + net.R, N0, [irr.t_irr])[0]
    atoms[~on] = propagate(net.A, eoi, times[~on] - irr.t_irr)
    return net, atoms


# (Z, A) of the nuclides with a third isomer ('O') in isotopia/files/decay: their
# ground state gets the third isomer's half-life in decaydata.f90 (lines 73-75).
THIRD_ISOMER_DEFECT = {(30, 61), (47, 95)}


@needs_samples
def test_port_gives_zn61_its_third_isomer_half_life():
    """Documents the decaydata.f90 defect: Zn-61 ground state, 89.1 s in the library,
    gets the 0.13 s of its third isomer in the Fortran reference output."""
    library = DecayLibrary()
    act = (SAMPLES / "p-Ga000-Ge068" / "org" / "Zn061.act").read_text(encoding="utf-8")
    port_half_life = float(act.split("Half-life [s]:")[1].split()[0])
    third = library.get((30, 61, 3))
    assert third is not None and port_half_life == pytest.approx(third.half_life, rel=1e-6)
    assert library.get((30, 61, 0)).half_life / port_half_life > 100


@needs_samples
@pytest.mark.parametrize("case", sample_cases())
def test_full_mode_matches_port_where_nothing_is_fed(case, tmp_path, monkeypatch):
    """Products made only directly from the target follow Eq. 4.2 in both codes.

    Compared, with decay back into the target switched off (see
    _irradiate_without_target_feeding): product states that the full decay
    network does not feed and that equations.f90 does not feed either, plus
    the target when it is stable.
    Fed products differ by design (defects 1-4) and are covered by the
    closed-form tests above. Also skipped:
    * states with no decay-library entry. The port copies the ground state's
      half-life to such an isomer (decaydata.f90 line 132); the matrix uses
      library data only and warns (test_full_mode_stops_on_missing_decay_data);
    * nuclides whose port decay constant is not the library's. The only
      known cause is a third isomer (letter O in the library), which
      decaydata.f90 lines 73-75 store as the ground state; any other
      disagreement fails the test.
    """
    library = DecayLibrary()
    compared, mismatches = 0, []

    def capture(st):
        nonlocal compared
        irr = Irradiation.from_state(st)
        net, atoms = _irradiate_without_target_feeding(irr, isotopia_times(st), library)
        fed = _fed_nuclides(net, irr.target)
        feed_irr, feed_cool = port_feeders(st)
        for n in [*irr.rates, irr.target]:
            z, a, i = n
            # Port keys: -1 holds a nuclide without isomers, and always the target.
            key = (z, a, i if st.Nisomer[(z, a)] >= 0 and n != irr.target else -1)
            if n == irr.target:
                if net.lam[net.index[n]] > 0:
                    continue
            elif n in fed or feed_irr.get(key) or feed_cool.get(key) or library.get(n) is None:
                continue
            lam_port, lam_lib = float(st.lam[key]), library.get(n).lam
            if lam_port != pytest.approx(lam_lib, rel=1e-6, abs=1e-17 * np.log(2)):
                if n[:2] not in THIRD_ISOMER_DEFECT:
                    mismatches.append((case, label(n), f"lambda {lam_port} vs {lam_lib}"))
                continue
            port = np.asarray(st.Niso[key], dtype=float)
            ok = np.abs(port) > FLOOR
            compared += 1
            mine = atoms[:, net.index[n]]
            if not np.allclose(mine[ok], port[ok], rtol=RTOL, atol=0):
                worst = np.max(np.abs(mine[ok] / port[ok] - 1))
                mismatches.append((case, label(n), worst))

    run_sample_capturing(case, tmp_path, monkeypatch, capture)
    assert compared > 0
    assert not mismatches, mismatches


@needs_samples
def test_full_mode_warnings_mo100(tmp_path, monkeypatch):
    found = []
    run_sample_capturing("p-Mo100-Tc099m", tmp_path, monkeypatch,
                         lambda st: found.append(Irradiation.from_state(st)))
    text = " ".join(found[0].warnings)
    assert "leaving Mo100 unchanged" in text          # (p,p') back to Mo-100
    assert "Tc99" in text and "state rates are used" in text
    assert "of R_T" in text and "not conserved" in text


@needs_samples
def test_full_mode_stops_on_missing_decay_data(tmp_path, monkeypatch):
    """g + Ra-226 makes Fr-224m, which has no entry in the decay library."""
    found = []
    run_sample_capturing("g-Ra226-Ac225", tmp_path, monkeypatch,
                         lambda st: found.append((Irradiation.from_state(st), isotopia_times(st))))
    irr, times = found[0]
    with pytest.raises(DecayDataError, match="Fr224_m1"):
        irradiate(irr, times)
    inv = irradiate(irr, times, on_missing="warn")
    assert any("Truncated chain" in w and "Fr224_m1" in w for w in inv.warnings)
