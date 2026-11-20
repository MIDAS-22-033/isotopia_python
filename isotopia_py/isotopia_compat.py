"""ISOTOPIA-compatible mode: equations.f90 rewritten as one matrix exponential.

This reproduces what the port's ``kinetics.equations`` (equations.f90) computes,
known defects included, so the matrix machinery can be checked against the
port nuclide by nuclide. It is a validation tool, not a physics model: for
real calculations use ``irradiation.irradiate``.

What equations.f90 does, and how each piece becomes a matrix term:

1. Target (lines 152-162): N_T = N_0 exp(-R_T t) during irradiation, then
   constant. Matrix: -R_T on the target diagonal; zero in cooling.
2. Direct production (lines 164-177): each product i gains R_T->i N_T and
   decays with lambda_i. Matrix: R_T->i in column T, -lambda_i on the diagonal.
3. Decay feeding in irradiation (lines 184-221): every state p (-1 total,
   0, 1, ...) of the nuclide at Z-1 whose first decay mode is beta-minus, or
   at Z+1 with beta-plus/EC, feeds every state of i (-1, 0, 1, ...) with its
   full lambda_p, and only if R_T->p > 0 and lambda_p > 0. The analytic
   formula uses p's *direct* production only, not p's own feeding. Matrix:
   a separate "shadow" compartment D_p, made from the target at R_T->p and
   decaying with lambda_p, which feeds i at lambda_p. The total (-1) and the
   ground state (0) of the same parent both feed i (defect 2).
4. Cooling (lines 178-183, 222-235): N_i decays from N_i(t_irr), and a parent
   p adds lambda_p N_p(t_irr) exp(-lambda_p tau) feeding. N_p(t_irr) is read
   from the stored inventory, which is zero for parents that equations.f90
   has not reached yet in its Z-descending loop, i.e. all beta-minus parents
   (defect 4). Matrix: shadow S_p starting at N_p(t_irr) if p comes earlier
   in the loop, else 0, decaying with lambda_p and feeding i.

Each phase is linear with constant coefficients, so exp(M t) gives the exact
value of the same equations. The port evaluates them in double precision
from float32 inputs and stores float32, so agreement is ~1e-7, except where
the port's closed form loses digits (defect 6, e.g. Ru-99 fed by Tc-99).

Inputs are taken from the port's State (rates, decay constants, decay types,
time grid), so both sides use exactly the same numbers.
"""
from __future__ import annotations

import numpy as np

from .irradiation import PortKey, isotopia_times, product_keys
from .matrix import propagate


def _beta_parents(st, key: PortKey):
    """Parent states that equations.f90 lets feed ``key`` (lines 187-194)."""
    iz, ia, _ = key
    for isob, mode in ((-1, 1), (1, 2)):          # Z-1 with beta-minus, Z+1 with beta-plus/EC
        zp = iz + isob
        if zp < 0:
            continue
        for is_p in range(-1, st.Nisomer[(zp, ia)] + 1):
            if st.rtyp[(zp, ia, is_p)] == mode:
                yield (zp, ia, is_p)


def port_feeders(st) -> tuple[dict[PortKey, list[PortKey]], dict[PortKey, list[PortKey]]]:
    """Parents that feed each product in equations.f90, in irradiation and in cooling.

    Parameters
    ----------
    st : State
        Port state after ``kinetics.equations``.

    Returns
    -------
    (dict, dict)
        For each product key, its feeding parent keys during irradiation
        (lines 194-199: decay type matches, R_T->p > 0 and lambda_p > 0) and
        during cooling (lines 222-224: parent already computed in the loop).
    """
    keys = product_keys(st)
    position = {k: n for n, k in enumerate(keys)}
    target = (st.Ztarget, st.Atarget, -1)
    feed_irr: dict[PortKey, list[PortKey]] = {k: [] for k in keys if k != target}
    feed_cool: dict[PortKey, list[PortKey]] = {k: [] for k in keys if k != target}
    if st.flagdecay:
        for i, parents in feed_irr.items():
            for p in _beta_parents(st, i):
                if float(st.reaction_rate[p]) > 0.0 and float(st.lam[p]) > 0.0:
                    parents.append(p)
                if p in position and position[p] < position[i]:
                    feed_cool[i].append(p)
    return feed_irr, feed_cool


def compat_inventory(st) -> dict[PortKey, np.ndarray]:
    """Atoms of every product on ISOTOPIA's time grid, as equations.f90 computes them.

    Parameters
    ----------
    st : State
        Port state right after ``kinetics.equations`` for one target isotope:
        uses N_0 [atoms], reaction_rate and lam [s^-1], rtyp, Nisomer,
        rpexist, Tgrid, Tir and Tco [s].

    Returns
    -------
    dict of (Z, A, is) to numpy.ndarray
        Atoms at each of the 101 grid times, keyed like the port's ``Niso``
        (is = -1 for totals). Compare with Niso before ``kinetics.activities``,
        which later overwrites the totals of nuclides with isomers.
    """
    if float(st.Tco) <= 0.0:
        raise ValueError("compatibility mode needs a cooling time > 0 (Tcool)")
    keys = product_keys(st)
    position = {k: n for n, k in enumerate(keys)}
    target = (st.Ztarget, st.Atarget, -1)
    products = [k for k in keys if k != target]
    R_T, N_0, t_irr = float(st.reaction_rate[(0, 0, -1)]), float(st.N_0), float(st.Tir)
    times = isotopia_times(st)
    n_irr = st.Ntime

    feed_irr, feed_cool = port_feeders(st)

    def rate(k):
        return float(st.reaction_rate[k])

    def lam(k):
        return float(st.lam[k])

    # Vector layout for both phases: [products..., shadow parents..., target].
    n = len(products)
    col = {k: j for j, k in enumerate(products)}
    out = {k: np.zeros(len(times)) for k in keys}

    # --- Irradiation: target, direct production, shadows D_p of direct-only parents
    shadows = sorted({p for ps in feed_irr.values() for p in ps})
    sh = {p: n + j for j, p in enumerate(shadows)}
    T = n + len(shadows)
    M = np.zeros((T + 1, T + 1))
    M[T, T] = -R_T
    for i in products:
        M[col[i], col[i]] = -lam(i)
        M[col[i], T] += rate(i)
        for p in feed_irr[i]:
            M[col[i], sh[p]] += lam(p)
    for p in shadows:
        M[sh[p], sh[p]] = -lam(p)
        M[sh[p], T] += rate(p)
    x0 = np.zeros(T + 1)
    x0[T] = N_0
    irr = propagate(M, x0, times[:n_irr + 1])
    for i in products:
        out[i][:n_irr + 1] = irr[:, col[i]]
    if target in position:
        out[target][:n_irr + 1] = irr[:, T]

    # --- Cooling: decay from N(t_irr), shadows S_p hold the stored parent inventory
    shadows = sorted({p for ps in feed_cool.values() for p in ps})
    sh = {p: n + j for j, p in enumerate(shadows)}
    T = n + len(shadows)
    M = np.zeros((T + 1, T + 1))                  # target stays constant: zero row
    for i in products:
        M[col[i], col[i]] = -lam(i)
        for p in feed_cool[i]:
            M[col[i], sh[p]] += lam(p)
    for p in shadows:
        M[sh[p], sh[p]] = -lam(p)
    x0 = np.zeros(T + 1)
    x0[:n] = irr[-1, :n]
    x0[T] = irr[-1, -1]
    for p in shadows:
        x0[sh[p]] = out[p][n_irr]
    cool = propagate(M, x0, times[n_irr + 1:] - t_irr)
    for i in products:
        out[i][n_irr + 1:] = cool[:, col[i]]
    if target in position:
        out[target][n_irr + 1:] = cool[:, T]
    return out
