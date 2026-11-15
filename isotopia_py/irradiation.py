"""Irradiation of a target with the matrix solver (stage 2: target reactions only).

The beam hits the target; the products decay through the full decay network
(every branch, isomeric transitions, chains of any length), but are not hit
again. Two phases (manual Section 4.1, Eq. 4.1):

    irradiation, 0 <= t <= t_irr:  dN/dt = (A + R) N,  N(0) = N_0 target atoms
    cooling,     t > t_irr:        dN/dt = A N,        starting from N(t_irr)

R holds the target terms of manual Eq. 4.2: the target depletes at its total
non-elastic rate R_T and feeds each product i at rate R_T->i. A is the decay
matrix built from the library (decaylib.py). Both phases are solved exactly
with the matrix exponential (matrix.py).

The beam, target and reaction rates come from a run of the Python port:
``Irradiation.from_state`` reads them from its State after ``run``. This keeps
the stopping power, beam spectrum and cross sections identical to ISOTOPIA,
so only the kinetics differ.

Physics that differs from ISOTOPIA (and so from the port):
* Products are fed by every decay branch with its branching ratio, including
  isomeric transitions and beta-minus feeding during cooling (defects 1-4 in
  claude/equations_f90_findings.md).
* An unstable target decays during irradiation and cooling; ISOTOPIA keeps it
  constant after irradiation and ignores its decay.
* Isomers use their own rates, R_T->g and R_T->m, also for the total.
  ISOTOPIA computes the total from a separate total rate, which in the
  library need not equal R_T->g + R_T->m.

Assumptions kept from ISOTOPIA:
* Constant beam, flux and target composition during irradiation: R does not
  change as the target depletes (burnup is a few 1e-3 or less in the samples).
* The target depletes at R_T, the full non-elastic rate, while reactions that
  leave the target unchanged (e.g. inelastic scattering to its ground state)
  are not counted as products. Those atoms therefore leave the bookkeeping;
  ``from_state`` warns with the size of this and any other imbalance.
* Only products inside the port's window (Zdepth, Adepth) are tracked.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .decaylib import DecayLibrary, Nuclide, label
from .matrix import Inventory, Reactions, build_chain, propagate

# Report rate inconsistencies larger than this (relative).
RATE_TOL = 1e-3

PortKey = tuple[int, int, int]   # ISOTOPIA's (Z, A, is) with is = -1 for the total


def product_keys(st) -> list[PortKey]:
    """Product keys in the order equations.f90 computes them (lines 123-130).

    Z from Zcomp+1 down by Zdepth, A from Acomp down by Adepth, isomer state
    -1 (total) to Nisomer. Keys without a residual cross section are skipped,
    except in the first row Z = Zcomp+1.

    Parameters
    ----------
    st : State
        Port state after ``run`` (or after ``kinetics.equations``).

    Returns
    -------
    list of (Z, A, is)
        ISOTOPIA keys, is = -1 for the total of a nuclide.
    """
    keys = []
    for iz in range(st.Zcomp + 1, st.Zcomp - st.Zdepth, -1):
        for ia in range(st.Acomp, st.Acomp - st.Adepth - 1, -1):
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                if iz == st.Zcomp + 1 or st.rpexist[(iz, ia, is_)]:
                    keys.append((iz, ia, is_))
    return keys


def isotopia_times(st) -> np.ndarray:
    """ISOTOPIA's output times [s]: 50 in irradiation, 50 in cooling, plus t = 0.

    Parameters
    ----------
    st : State
        Port state after ``run``.

    Returns
    -------
    numpy.ndarray
        Times [s], shape (101,), as the port's float32 grid converted to float64.
    """
    return np.array([float(t) for t in st.Tgrid])


@dataclass(frozen=True)
class Irradiation:
    """A target, its beam-induced reaction rates and the irradiation times.

    Attributes
    ----------
    target : Nuclide
        Target nuclide (Z, A, I).
    N0 : float
        Target atoms at the start of irradiation.
    burnup : float
        R_T, total non-elastic reaction rate per target atom [s^-1].
    rates : dict of Nuclide to float
        R_T->i, production rate of each product state per target atom [s^-1].
    t_irr : float
        Irradiation time [s].
    t_cool : float
        Cooling time [s].
    warnings : tuple of str
        Data inconsistencies found while reading the rates.
    """

    target: Nuclide
    N0: float
    burnup: float
    rates: dict[Nuclide, float]
    t_irr: float
    t_cool: float
    warnings: tuple[str, ...] = field(default=())

    @classmethod
    def from_state(cls, st) -> Irradiation:
        """Read target, rates and times from a port run (one target isotope).

        For a natural target, ``run`` loops over the isotopes and the State
        holds only the last one; call this inside that loop instead.

        Parameters
        ----------
        st : State
            Port state after ``run``: uses N_0 [atoms], reaction_rate [s^-1],
            Tir and Tco [s], Nisomer, rpexist and the depth window.

        Returns
        -------
        Irradiation
            With state-resolved product rates keyed (Z, A, I).
        """
        target = (st.Ztarget, st.Atarget, 0)
        R_T = float(st.reaction_rate[(0, 0, -1)])
        rates: dict[Nuclide, float] = {}
        warnings: list[str] = []
        self_rate = 0.0

        # Map ISOTOPIA keys to states: nuclides with isomers use is = 0, 1, ...;
        # the others only have the total (is = -1), which is their ground state.
        for z, a, is_ in product_keys(st):
            has_isomers = st.Nisomer[(z, a)] >= 0
            if (is_ == -1) == has_isomers:
                continue
            # The library's total and its isomer states need not agree.
            if has_isomers and is_ == st.Nisomer[(z, a)]:
                total = float(st.reaction_rate[(z, a, -1)])
                states = sum(float(st.reaction_rate[(z, a, j)]) for j in range(is_ + 1))
                if total > 0 and abs(states / total - 1) > RATE_TOL:
                    warnings.append(f"Rates to the states of {label((z, a, 0))} sum to "
                                    f"{states:.6e} s^-1 but its total is {total:.6e} s^-1; "
                                    "the state rates are used.")
            rate = float(st.reaction_rate[(z, a, is_)])
            key = (z, a, max(is_, 0))
            if rate <= 0.0:
                continue
            if key == target:
                self_rate = rate
                continue
            rates[key] = rate

        # Atom balance: the target loses R_T, the products gain the sum of R_T->i.
        produced = sum(rates.values())
        if self_rate > 0 and R_T > 0:
            warnings.append(f"Reactions leaving {label(target)} unchanged (R = {self_rate:.6e} "
                            f"s^-1, {self_rate / R_T:.4f} of R_T) are not products; the target "
                            "still depletes at the full R_T, as in ISOTOPIA (manual Eq. 4.2).")
        if R_T > 0 and abs(produced / R_T - 1) > RATE_TOL:
            warnings.append(f"Product rates sum to {produced / R_T:.4f} of R_T = {R_T:.6e} s^-1: "
                            "atoms are not conserved during irradiation by this fraction.")
        return cls(target, float(st.N_0), R_T, rates, float(st.Tir), float(st.Tco),
                   tuple(warnings))


def irradiate(irr: Irradiation, times: Iterable[float],
              library: DecayLibrary | None = None, on_missing: str = "error") -> Inventory:
    """Inventories during irradiation and cooling, from the matrix solver.

    Parameters
    ----------
    irr : Irradiation
        Target atoms, reaction rates [s^-1] and times [s].
    times : iterable of float
        Output times [s] from the start of irradiation, each >= 0. Times up to
        ``irr.t_irr`` are in irradiation, later ones in cooling.
    library : DecayLibrary, optional
        Decay data. Default: the ISOTOPIA decay library (see decaylib).
    on_missing : {'error', 'warn'}
        Handling of nuclides absent from the decay library; see
        ``matrix.build_chain``.

    Returns
    -------
    Inventory
        Atoms of every nuclide at each time, with warnings.
    """
    library = library or DecayLibrary()
    net = build_chain([irr.target, *irr.rates], library, on_missing,
                      reactions={irr.target: Reactions(irr.burnup, irr.rates)})
    N0 = np.zeros(len(net.nuclides))
    N0[net.index[irr.target]] = irr.N0

    # Irradiation phase with M = A + R, then cooling with M = A from N(t_irr).
    times = np.asarray(list(times), dtype=float)
    beam_on = times <= irr.t_irr
    atoms = np.empty((len(times), len(N0)))
    atoms[beam_on] = propagate(net.A + net.R, N0, times[beam_on])
    if (~beam_on).any():
        eoi = propagate(net.A + net.R, N0, [irr.t_irr])[0]
        atoms[~beam_on] = propagate(net.A, eoi, times[~beam_on] - irr.t_irr)
    return Inventory(times, net.nuclides, atoms, net.lam, [*irr.warnings, *net.warnings])
