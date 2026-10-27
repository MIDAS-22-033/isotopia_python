"""Production, depletion and decay of nuclides (equations.f90, activities.f90).

During irradiation the target depletes as N_T(t) = N_0 exp(-R_T t), where R_T
is the total non-elastic reaction rate per target atom. Each product i is made
directly from the target with rate constant R_i and decays with lambda_i:

    N_i(t) = N_0 R_i (exp(-R_T t) - exp(-lambda_i t)) / (lambda_i - R_T)

A product can also be fed by beta decay of a parent p (beta- from Z-1, beta+
from Z+1) that is itself produced from the target. After irradiation the
inventories decay, with parents feeding daughters through the two-member
Bateman solution.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import AVOGADRO, DAYSEC, HOURSEC, MINUTESEC, NUMTIME, YEARSEC
from .fortran import S, fexp, time_breakdown
from .state import State


def equations(st: State) -> None:
    """Number of nuclides of each product on the time grid."""
    with np.errstate(all="ignore"):
        _equations(st)


def _equations(st: State) -> None:
    # Initial condition for irradiation
    st.N_0 = AVOGADRO / S(st.Atarget) * st.M_target
    if st.iso == 1:
        st.N_0_nat = st.abun[1] * st.N_0
    else:
        st.N_0_nat = st.N_0_nat + st.abun[st.iso] * st.N_0

    # Time grid: Ntime points up to the end of irradiation, the rest for cooling
    st.Ntime = NUMTIME // 2
    Ntime = st.Ntime
    Tgrid = st.Tgrid
    dT = st.Tir / S(Ntime)
    for it in range(0, Ntime + 1):
        Tgrid[it] = S(it) * dT
    if st.Tco > 0:
        Ncool = NUMTIME - Ntime
        dT = st.Tco / S(Ncool)
        for it in range(Ntime + 1, NUMTIME + 1):
            Tgrid[it] = st.Tir + S(it - Ntime) * dT
    Tir = float(st.Tir)
    N_0 = float(st.N_0)

    # Fixed relative to equations.f90: start every isotope run of a natural
    # target from empty inventories. The Fortran only zeroes the nuclides it
    # is about to compute, so a nuclide produced from one isotope but absent
    # from the next kept its old inventory, which activities.f90 then added
    # to the natural-element totals with the new isotope's abundance.
    st.Niso.clear()
    st.Nisorel.clear()

    R_T = float(st.reaction_rate[(0, 0, -1)])
    Zcomp, Acomp = st.Zcomp, st.Acomp
    for iz in range(Zcomp + 1, Zcomp + 1 - st.Zdepth - 1, -1):
        Nisotot = st.Nisotot[iz]
        Nisotot[:] = 0
        for ia in range(Acomp, Acomp - st.Adepth - 1, -1):
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                key = (iz, ia, is_)
                st.Yexist[key] = False
                if iz != Zcomp + 1 and not st.rpexist[key]:
                    continue
                st.Tmax[key] = S(0)
                st.Tmaxactivity[key] = 0
                st.Tp[key] = [0] * 6
                if st.iso == 1:
                    st.Ynatexist[key] = False
                Niso = st.Niso[key]
                Niso[:] = 0
                st.Nisorel[key][:] = 0
                is_target = iz == st.Ztarget and ia == st.Atarget and is_ == -1
                if is_target:
                    Niso[0] = st.N_0
                    Nisotot[0] = Nisotot[0] + Niso[0]
                R_Ti = float(st.reaction_rate[key])
                lambda_i = float(st.lam[key])
                enum_i = N_0 * R_Ti
                denom_i = lambda_i - R_T
                for it in range(1, NUMTIME + 1):
                    TT = float(Tgrid[it])
                    # Depletion of target
                    if is_target:
                        if it <= Ntime:
                            Niso[it] = N_0 * fexp(-R_T * TT)
                        else:
                            Niso[it] = Niso[Ntime]
                    else:
                        # 1. Production directly from target
                        exp_T = fexp(-R_T * TT)
                        exp_i = fexp(-lambda_i * TT)
                        if it <= Ntime:
                            eps_decay = 1.0e-12 * max(abs(lambda_i), abs(R_T), 1.0e-30)
                            if abs(denom_i) > eps_decay:
                                term_Ti = (exp_T - exp_i) / denom_i
                            else:
                                term_Ti = TT * exp_i
                            Niso[it] = enum_i * term_Ti
                        else:
                            # Cooling
                            exp_c = fexp(-lambda_i * (TT - Tir))
                            Niso[it] = float(Niso[Ntime]) * exp_c
                        # 2. Production from decay of another nuclide
                        if st.flagdecay:
                            for isob in (-1, 1):
                                Zparent = iz + isob
                                if Zparent < 0:
                                    continue
                                # Account for isomeric decay of parent
                                for is_p in range(-1, st.Nisomer[(Zparent, ia)] + 1):
                                    pkey = (Zparent, ia, is_p)
                                    rt = st.rtyp[pkey]
                                    if not ((isob == -1 and rt == 1) or (isob == 1 and rt == 2)):
                                        continue
                                    lambda_p = float(st.lam[pkey])
                                    denom_ip = lambda_i - lambda_p
                                    if it <= Ntime:
                                        R_Tp = float(st.reaction_rate[pkey])
                                        if R_Tp > 0.0 and lambda_p > 0.0:
                                            exp_p = fexp(-lambda_p * TT)
                                            eps_decay = 1.0e-12 * max(abs(lambda_i), abs(lambda_p), 1.0e-30)
                                            if abs(denom_ip) > eps_decay:
                                                term_pi = (exp_p - exp_i) / denom_ip
                                            else:
                                                term_pi = TT * exp_p
                                            enum_p = N_0 * R_Tp
                                            denom_p = lambda_p - R_T
                                            if abs(denom_p) > eps_decay:
                                                term = lambda_p * enum_p / denom_p * (term_Ti - term_pi)
                                            elif abs(denom_i) > eps_decay:
                                                term = (R_T * enum_p * (TT * exp_T * denom_i - exp_T + exp_i)
                                                        / (denom_i * denom_i))
                                            else:
                                                term = 0.5 * R_T * enum_p * TT * TT * exp_T
                                            Niso[it] = float(Niso[it]) + term
                                    else:
                                        # Cooling only
                                        N_p = float(st.Niso[pkey][Ntime])
                                        exp_cp = fexp(-lambda_p * (TT - Tir))
                                        exp_ci = fexp(-lambda_i * (TT - Tir))
                                        tau = TT - Tir
                                        eps_decay = 1.0e-12 * max(abs(lambda_i), abs(lambda_p), 1.0e-30)
                                        if abs(denom_ip) > eps_decay:
                                            term = N_p * lambda_p / denom_ip * (exp_cp - exp_ci)
                                        else:
                                            term = N_p * lambda_p * tau * exp_cp
                                        Niso[it] = float(Niso[it]) + term
                    if Niso[it] > 0:
                        st.Yexist[key] = True
                        st.Ynatexist[key] = True
                    Nisotot[it] = Nisotot[it] + Niso[it]
                if not st.Yexist[key]:
                    continue
                if lambda_i > 0.0 and R_T > 0.0:
                    st.Tmax[key] = S(math.log(lambda_i / R_T) / (lambda_i - R_T))
                else:
                    st.Tmax[key] = S(1.0e30)
                # Irradiation time with maximum yield in years, days, etc.
                st.Tp[key] = [0] + time_breakdown(float(st.Tmax[key]), YEARSEC, DAYSEC, HOURSEC,
                                                  MINUTESEC, single=False)
        for ia in range(Acomp, Acomp - st.Adepth - 1, -1):
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                key = (iz, ia, is_)
                if st.Yexist[key]:
                    Niso = st.Niso[key]
                    Nisorel = st.Nisorel[key]
                    for it in range(0, NUMTIME + 1):
                        if Nisotot[it] != 0:
                            Nisorel[it] = Niso[it] / Nisotot[it]


def activities(st: State) -> None:
    """Activities, yields and specific activities from decay constants."""
    with np.errstate(all="ignore"):
        _activities(st)


def _activities(st: State) -> None:
    if st.k0 <= 1:
        activity_factor = st.M_target
    else:
        activity_factor = st.Ibeam
    st.activity.clear()
    st.yld.clear()
    st.specactivity.clear()
    if st.iso == 1:
        for arr in (st.Nisonat, st.activitynat, st.specactivitynat, st.yieldnat,
                    st.Nisorelnat, st.Nisototnat):
            arr.clear()
        st.reaction_ratenat.reset()
    Tgrid = st.Tgrid
    Ntime = st.Ntime
    Zcomp, Acomp = st.Zcomp, st.Acomp
    zrange = range(Zcomp + 1, Zcomp + 1 - st.Zdepth - 1, -1)
    arange = range(Acomp, Acomp - st.Adepth - 1, -1)
    for iz in zrange:
        for ia in arange:
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                key = (iz, ia, is_)
                lam = float(st.lam[key])
                act = st.activity[key]
                Niso = st.Niso[key]
                for it in range(1, NUMTIME + 1):
                    act[it] = lam * float(Niso[it])
                yld = st.yld[key]
                for it in range(1, Ntime + 1):
                    dA = act[it] - act[it - 1]
                    yld[it] = max(float(dA) / (float(activity_factor)
                                               * float(Tgrid[it] - Tgrid[it - 1])), 0.0)
    for iz in zrange:
        for ia in arange:
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                key = (iz, ia, is_)
                tot = (iz, ia, -1)
                acmax = S(0)
                act = st.activity[key]
                spec = st.specactivity[key]
                for it in range(1, NUMTIME + 1):
                    # MV: correct the total (is = -1) when an isomer is present. The
                    # condition on tmax guarantees that if the isomer activity goes to
                    # zero, the code is still executed.
                    itm = int(st.Tmaxactivity[tot])
                    if is_ == 1 and st.Niso[(iz, ia, 1)][itm] > 0:
                        st.Niso[tot][it] = st.Niso[(iz, ia, 0)][it] + st.Niso[(iz, ia, 1)][it]
                        st.activity[tot][it] = st.activity[(iz, ia, 0)][it] + st.activity[(iz, ia, 1)][it]
                    if is_ == 1 and st.activity[tot][it] > st.activity[tot][itm]:
                        st.Tmaxactivity[tot] = it
                    if act[it] > acmax:
                        st.Tmaxactivity[key] = it
                        acmax = act[it]
                    spec[it] = act[it] / st.M_target
    if st.flagnatural:
        ab = st.abun[st.iso]
        for iz in zrange:
            for ia in arange:
                for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                    key = (iz, ia, is_)
                    st.activitynat[key][1:] += ab * st.activity[key][1:]
                    st.specactivitynat[key][1:] += ab * st.specactivity[key][1:]
                    st.yieldnat[key][1:] += ab * st.yld[key][1:]
                    st.Nisonat[key][1:] += ab * st.Niso[key][1:]
                    st.Nisototnat[iz][1:] += ab * st.Niso[key][1:]
                    st.reaction_ratenat[key] = st.reaction_ratenat[key] + ab * st.reaction_rate[key]
                    if st.iso == st.isonum:
                        for it in range(1, NUMTIME + 1):
                            if st.Nisototnat[iz][it] != 0:
                                st.Nisorelnat[key][it] = st.Nisonat[key][it] / st.Nisototnat[iz][it]
                            st.Nelrel[iz][it] = st.Nisototnat[iz][it] / st.N_0_nat
