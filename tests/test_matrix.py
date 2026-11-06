"""Tests for the decay-only matrix solver (isotopia_py/matrix.py, decaylib.py).

1. Synthetic chains (made-up decay constants, no library) against closed-form
   Bateman solutions (manual Appendix A, Eqs. A.11-A.19 with R = 0).
2. Library reading and chains built from the JEFF-3.1.1 files in
   isotopia/files/decay (skipped when absent).
3. The cooling phase of sample p-Mo100-Tc099m: nuclides with no feeding must
   reproduce the Fortran; fed nuclides may only gain atoms (skipped when absent).
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest

from isotopia_py.constants import NUC
from isotopia_py.decaylib import (
    DecayDataError,
    DecayMode,
    NuclideDecay,
    default_decay_dir,
)
from isotopia_py.io import read_act
from isotopia_py.matrix import build_chain, decay

# ----------------------------------------------------------------------------
# 1. Synthetic chains
# ----------------------------------------------------------------------------


class FakeLibrary:
    """In-memory stand-in for DecayLibrary: {nuclide: (lambda, [(daughter, BR), ...])}."""

    def __init__(self, table):
        self.table = table

    def file_for(self, z):
        return Path(f"fake/z{z:03d}")

    def get(self, n):
        if n not in self.table:
            return None
        lam, modes = self.table[n]
        half = math.log(2) / lam if lam > 0 else math.inf
        return NuclideDecay(n, half, lam, tuple(DecayMode("1", d, br) for d, br in modes), "fake")


P, D, G, E = (1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)   # parent, daughter, granddaughter, end


def chain_lib(*lams):
    """Linear chain P -> D -> G -> ... with the given decay constants; last one is stable."""
    keys = [P, D, G, E][:len(lams) + 1]
    table = {k: (lam, [(keys[i + 1], 1.0)]) for i, (k, lam) in enumerate(zip(keys, lams))}
    table[keys[-1]] = (0.0, [])
    return FakeLibrary(table)


def test_single_decay():
    lam, t = 1e-4, np.array([0.0, 1e3, 1e4, 1e5])
    inv = decay({P: 1e20}, t, chain_lib(lam))
    np.testing.assert_allclose(inv[P], 1e20 * np.exp(-lam * t), rtol=1e-13)
    np.testing.assert_allclose(inv[P] + inv[D], 1e20, rtol=1e-14)


def bateman(lams, t):
    """Last member of a linear chain with distinct decay constants, N1(0) = 1.

    Classic Bateman sum, evaluated with 40-digit decimals: in float64 its
    terms cancel badly at short times (2e-12 error for the chain below).
    """
    getcontext().prec = 40
    L, t = [Decimal(repr(x)) for x in lams], Decimal(repr(float(t)))
    total = Decimal(0)
    for i, li in enumerate(L):
        den = Decimal(1)
        for j, lj in enumerate(L):
            if j != i:
                den *= lj - li
        total += (-li * t).exp() / den
    for lj in L[:-1]:
        total *= lj
    return float(total)


def test_three_member_bateman():
    lams = (3e-6, 3e-5, 1e-7)
    t = np.array([1728.0, 86400.0, 3.15e7])
    inv = decay({P: 1.0}, t, chain_lib(*lams))
    np.testing.assert_allclose(inv[D], [bateman(lams[:2], x) for x in t], rtol=1e-12)
    np.testing.assert_allclose(inv[G], [bateman(lams, x) for x in t], rtol=1e-12)


def test_equal_decay_constants():
    """The analytic form divides by (lambda_2 - lambda_1); the matrix has no such singularity."""
    lam, t = 2e-5, np.array([100.0, 86400.0, 1e6])
    inv = decay({P: 1.0}, t, chain_lib(lam, lam))
    np.testing.assert_allclose(inv[D], lam * t * np.exp(-lam * t), rtol=1e-12)


def test_stiff_short_lived_intermediate():
    """A 1e-6 s intermediate between slow members; the end product must still be exact."""
    l1, l2 = 1e-4, 1e6
    t = np.array([1.0, 1728.0, 86400.0])
    inv = decay({P: 1.0}, t, chain_lib(l1, l2))
    np.testing.assert_allclose(inv[P], np.exp(-l1 * t), rtol=1e-13)
    np.testing.assert_allclose(inv[D], l1 / (l2 - l1) * (np.exp(-l1 * t) - np.exp(-l2 * t)),
                               rtol=1e-6)   # D is ~1e-10 of the total: rounding floor
    exact_G = 1 - (l2 * np.exp(-l1 * t) - l1 * np.exp(-l2 * t)) / (l2 - l1)
    np.testing.assert_allclose(inv[G], exact_G, rtol=1e-12)


def test_branching_and_isomeric_transition():
    """P -> M (BR b) and P -> G (1 - b); M -> G by IT. Atoms are conserved."""
    M, b, lp, lm = (3, 1, 1), 0.88, 3e-6, 3e-5
    lib = FakeLibrary({P: (lp, [(M, b), (G, 1 - b)]), M: (lm, [(G, 1.0)]), G: (0.0, [])})
    t = np.array([1728.0, 86400.0, 1e7])
    inv = decay({P: 1.0}, t, lib)
    np.testing.assert_allclose(inv[M], b * lp / (lm - lp) * (np.exp(-lp * t) - np.exp(-lm * t)),
                               rtol=1e-12)
    np.testing.assert_allclose(inv[P] + inv[M] + inv[G], 1.0, rtol=1e-14)


def test_matrix_upper_triangular_daughters_first():
    lib = chain_lib(1e-3, 1e-5, 1e-2)
    chain = build_chain([P], lib)
    assert chain.nuclides == [E, G, D, P]
    assert np.allclose(chain.A, np.triu(chain.A))
    assert not chain.warnings


def test_time_zero_and_negative_times():
    inv = decay({P: 5.0}, [0.0], chain_lib(1e-3))
    assert inv[P][0] == 5.0 and inv[D][0] == 0.0
    with pytest.raises(ValueError):
        decay({P: 5.0}, [-1.0], chain_lib(1e-3))


def test_missing_data_stops_or_warns():
    lib = FakeLibrary({P: (1e-3, [(D, 1.0)])})   # D absent from the "library"
    with pytest.raises(DecayDataError, match="fake/z002"):
        decay({P: 1.0}, [10.0], lib)
    inv = decay({P: 1.0}, [1e4], lib, on_missing="warn")
    assert any("Truncated chain" in w for w in inv.warnings)
    np.testing.assert_allclose(inv[P] + inv[D], 1.0, rtol=1e-14)


def test_fission_and_branching_sum_warnings():
    lib = FakeLibrary({P: (1e-3, [(None, 0.1), (D, 0.85)]), D: (0.0, [])})
    inv = decay({P: 1.0}, [1e5], lib)
    assert any("spontaneous fission" in w for w in inv.warnings)
    assert any("sum to 0.95" in w for w in inv.warnings)
    np.testing.assert_allclose(inv[D], 0.85, rtol=1e-12)


# ----------------------------------------------------------------------------
# 2. Decay library
# ----------------------------------------------------------------------------

DECAY_DIR = default_decay_dir()
needs_library = pytest.mark.skipif(not DECAY_DIR.is_dir(), reason=f"no decay library at {DECAY_DIR}")
MO99, TC99M, TC99, RU99 = (42, 99, 0), (43, 99, 1), (43, 99, 0), (44, 99, 0)


@needs_library
def test_library_modes_mo99_tc99m():
    from isotopia_py.decaylib import DecayLibrary
    lib = DecayLibrary()
    mo = lib.get(MO99)
    assert {m.daughter for m in mo.modes} == {TC99, TC99M}
    assert sum(m.branching for m in mo.modes) == pytest.approx(1.0, abs=1e-3)
    assert mo.lam == pytest.approx(math.log(2) / mo.half_life, rel=1e-15)
    assert ("IT", TC99) in {(m.name, m.daughter) for m in lib.get(TC99M).modes}


@needs_library
def test_library_mo99_chain_against_bateman():
    """Tc-99m from Mo-99, with lambda and branching ratio taken from the library."""
    from isotopia_py.decaylib import DecayLibrary
    lib = DecayLibrary()
    lp, lm = lib.get(MO99).lam, lib.get(TC99M).lam
    b = sum(m.branching for m in lib.get(MO99).modes if m.daughter == TC99M)
    t = np.linspace(0.0, 1e6, 11)
    inv = decay({MO99: 1e15}, t, lib)
    assert set(inv.nuclides) == {MO99, TC99M, TC99, RU99} and not inv.warnings
    np.testing.assert_allclose(inv[TC99M], 1e15 * b * lp / (lm - lp)
                               * (np.exp(-lp * t) - np.exp(-lm * t)), rtol=1e-12)
    np.testing.assert_allclose(inv.atoms.sum(axis=1), 1e15, rtol=1e-13)


# ----------------------------------------------------------------------------
# 3. Cooling phase of sample p-Mo100-Tc099m
# ----------------------------------------------------------------------------

SAMPLE = DECAY_DIR.parents[1] / "samples" / "p-Mo100-Tc099m" / "org"
# Reference inventories and decay constants are float32 and printed to 7 digits;
# exp(-lambda t) then carries ~6e-8 * lambda t relative error. 1e-5 covers this.
RTOL = 1e-5
# float32 values near the underflow limit (1.2e-38) lose digits: compare above this.
FLOOR = 1e-30


def _reference_files():
    """Reference .act files per state-resolved nuclide (Z, A, I).

    Files without a g/m suffix are the ground state of a nuclide without
    isomers; for nuclides with isomers they are the total and are skipped.
    """
    symbol = {s.strip(): z + 1 for z, s in enumerate(NUC)}
    acts = {}
    for f in SAMPLE.glob("*.act"):
        m = re.fullmatch(r"([A-Z][a-z]?)(\d{3})([gm]?)\.act", f.name)
        acts[(symbol[m[1]], int(m[2]), m[3])] = read_act(f)
    return {(z, a, 1 if s == "m" else 0): act for (z, a, s), act in acts.items()
            if not (s == "" and (z, a, "g") in acts)}


def _fed(chain, initial):
    """Nuclides that receive atoms by decay of something present at t = 0.

    ``chain.A[i, j] != 0`` (i != j) means j feeds i. Spreading 'is fed' from
    the initially present nuclides down that graph marks every descendant.
    """
    feeds = (chain.A != 0) & ~np.eye(len(chain.nuclides), dtype=bool)
    fed = np.zeros(len(chain.nuclides), dtype=bool)
    sources = np.array([initial.get(n, 0.0) > 0 for n in chain.nuclides])
    while True:
        new = feeds[:, sources | fed].any(axis=1) | fed
        if (new == fed).all():
            return {n for n, f in zip(chain.nuclides, fed) if f}
        fed = new


@needs_library
@pytest.mark.skipif(not SAMPLE.is_dir(), reason=f"no reference sample at {SAMPLE}")
def test_cooling_against_fortran_mo100_sample():
    from isotopia_py.decaylib import DecayLibrary
    ref = _reference_files()
    t = np.array(ref[TC99M].column("time_s"))
    eoi = int(np.argmax(t >= 86400.0))   # end of irradiation (Tirrad 1 d)
    initial = {k: act.column("isotope_inventory")[eoi] for k, act in ref.items()}
    lib = DecayLibrary()
    inv = decay(initial, t[eoi:] - t[eoi], lib)
    fed = _fed(build_chain(initial, lib), initial)
    assert not inv.warnings
    for k, act in ref.items():
        R = np.array(act.column("isotope_inventory"))[eoi:]
        M = inv[k]
        ok = R > FLOOR
        # Same decay constant as the Fortran (float32). decaydata.f90 sets lambda = 0
        # for half-lives above 1e17 s; the matrix keeps the library value.
        lam_ref = act.metadata.get("Decay constant [s^-1]")
        if lam_ref is not None:
            lam_ref, lam = float(lam_ref), inv.lam[inv.nuclides.index(k)]
            if lam_ref == 0.0:
                assert lam < math.log(2) / 1e17, k
            else:
                assert lam == pytest.approx(lam_ref, rel=1e-6), k
        if k in fed:
            # The Fortran misses isomeric transitions and beta-minus feeding in cooling,
            # so the full decay network can only add atoms here.
            assert np.all(M[ok] >= R[ok] * (1 - RTOL)), k
        else:
            np.testing.assert_allclose(M[ok], R[ok], rtol=RTOL, err_msg=str(k))
