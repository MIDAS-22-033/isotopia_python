"""Beam spectra, stopping power and reaction rates (stoppingpower.f90, reactionrates.f90)."""
from __future__ import annotations

import os

import numpy as np

from . import yandf
from .constants import AMU, AVOGADRO, EMASS, PARMASS, PARZ, QELEM
from .data import crosssections
from .fortran import S, es, i, list_reals, locate, pol1, sexp, slog, spow
from .state import State


def stoppingpower(st: State, E) -> np.float32:
    """Bethe-Bloch stopping power dE/dx in MeV/cm (Leo, 1994, p. 24)."""
    k0 = st.k0
    E = S(E)
    pmass = S(PARMASS[k0] * AMU)
    enum = E * (E + S(2) * pmass)
    denom = (E + pmass) * (E + pmass)
    beta = np.sqrt(enum / denom)
    Z = S(st.Ztarget)
    Imean = Z * (S(9.76) + S(58.8) * spow(Z, S(-1.19))) * S(1e-6)
    gam = S(1) / np.sqrt(S(1) - beta * beta)
    eta = beta * gam
    Wmax = S(2) * EMASS * (eta * eta)
    term1 = (S(0.307075) * st.rho_target * Z / S(st.Atarget) * S(PARZ[k0] ** 2)
             / (beta * beta))
    term2 = S(0.5) * slog((Wmax * Wmax) / (Imean * Imean)) - beta * beta
    return term1 * term2


def _read_photon_spectrum(st: State):
    iE = max(min(200, int(st.Ebeam)), 1)
    specfile = st.path + "files/photon_spectra/brems_" + i(iE, 3, 3) + "MeV.txt"
    Eint, phi = [], []
    with open(specfile, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw[:1] == "#":
                continue
            e, p = list_reals(raw, 2)
            Eint.append(S(e))
            phi.append(S(p))
    return Eint, phi


def _read_neutron_spectrum(st: State):
    specfile = st.path + "files/neutron_spectra/n-hfr.dat"
    Eint, dEint, phi = [], [], []
    with open(specfile, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw[:1] == "#":
                continue
            _, Ea, Eb, _, p = list_reals(raw, 5)
            Ea, Eb = S(Ea), S(Eb)
            Eint.append(S(0.5) * (Ea + Eb) * S(1e-6))
            dEint.append(np.abs(Ea - Eb) * S(1e-6))
            phi.append(S(p))
    return Eint, dEint, phi


def reactionrates(st: State) -> None:
    with np.errstate(all="ignore"):
        _reactionrates(st)


def _reactionrates(st: State) -> None:
    k0 = st.k0
    projrate_density = S(0)
    st.M_target = st.targetmass
    st.Eaverage = S(0)
    # Arrays are 1-based; index 0 is unused.
    Eint = [S(0)]
    dEint = [S(0)]
    phi = [S(0)]

    # Photons: bremsstrahlung spectrum for the incident electron energy
    if k0 == 0:
        e, p = _read_photon_spectrum(st)
        Eint += e
        phi += p
        NintE = len(e)
        dEint = [S(0)] * (NintE + 1)
        norm = S(0)
        for nE in range(1, NintE + 1):
            if nE == 1:
                dEint[nE] = S(0.5) * (Eint[2] - Eint[1])
            elif nE == NintE:
                dEint[nE] = S(0.5) * (Eint[NintE] - Eint[NintE - 1])
            else:
                dEint[nE] = S(0.5) * (Eint[nE + 1] - Eint[nE - 1])
            norm = norm + phi[nE] * dEint[nE]
            st.Eaverage = st.Eaverage + Eint[nE] * phi[nE] * dEint[nE]
        if norm > 0:
            st.Eaverage = st.Eaverage / norm
            for nE in range(1, NintE + 1):
                phi[nE] = phi[nE] / norm

    # Neutrons: reactor spectrum
    if k0 == 1:
        e, de, p = _read_neutron_spectrum(st)
        Eint += e
        dEint += de
        phi += p
        NintE = len(e)
        for nE in range(1, NintE + 1):
            st.Eaverage = st.Eaverage + Eint[nE] * phi[nE]

    # Charged particles: integration grid and stopping power
    if k0 > 1:
        NintE = 100
        dE = (st.Ebeam - st.Eback) / S(NintE)
        phisum = S(0)
        Eint = [S(0)] * (NintE + 1)
        dEint = [S(0)] * (NintE + 1)
        phi = [S(0)] * (NintE + 1)
        for nE in range(1, NintE + 1):
            Eint[nE] = st.Eback + (S(nE) - S(0.5)) * dE
            dEint[nE] = dE
            Sp = stoppingpower(st, Eint[nE])
            if Sp != 0:
                phi[nE] = S(1) / Sp
            phisum = phisum + phi[nE] * dE
            st.Eaverage = st.Eaverage + Eint[nE] * phi[nE] * dEint[nE]
        st.Leff = phisum
        st.V_target = st.Area * st.Leff
        # Ibeam is in A, energies are in MeV, heat is in kW
        st.heat = S(1000) * st.Ibeam * (st.Ebeam - st.Eback) / S(PARZ[k0])
        st.projnum = st.Ibeam / (S(PARZ[k0]) * QELEM)
        projrate_density = st.projnum / st.V_target
        st.Eaverage = st.Eaverage / st.Leff
    if k0 == 0:
        st.projnum = st.Ibeam / QELEM
        st.fluxtotal = st.projnum * st.fgamma / st.Area
    if k0 <= 1:
        if st.targetmass_input != S(-1):
            st.V_target = st.targetmass / st.rho_target
            st.thickness = st.V_target / st.Area
        else:
            st.V_target = st.Area * st.thickness
    st.M_target = st.rho_target * st.V_target

    # Reaction rates
    st.Nisomer.reset()
    st.rpexist.reset()
    st.reaction_rate.reset()
    st.sacs.reset()
    G = [S(1)] * (NintE + 1)
    st.selfshield_av = S(1)
    number_density = AVOGADRO / S(st.Atarget) * st.rho_target
    crosssections(st, 0, 0, -1)

    # Self-shielding
    if st.flagselfshield and k0 == 1:
        st.selfshield_av = S(0)
        for nE in range(1, NintE + 1):
            E = Eint[nE]
            nen = locate(st.Etot, 1, st.Nentot, E)
            if nen == 0:
                continue
            xst = pol1(st.Etot[nen], st.Etot[nen + 1], st.xstot[nen], st.xstot[nen + 1], E)
            xsmacro = number_density * xst * S(1e-27)
            term = xsmacro * st.thickness
            if term > 0:
                G[nE] = (S(1) - sexp(-term)) / term
            st.selfshield_av = st.selfshield_av + G[nE] * phi[nE]

    # Loop over residual nuclides, then the total non-elastic (0, 0, -1)
    keys = [(iz, ia, is_)
            for iz in range(st.Zcomp + 1, st.Zcomp - st.Zdepth - 1, -1)
            for ia in range(st.Acomp, st.Acomp - st.Adepth - 1, -1)
            for is_ in (-1, 0, 1, 2)]
    keys.append((0, 0, -1))
    for key in keys:
        if key == (0, 0, -1):
            st.Nenrp = st.Nennon
            st.xsrp = st.xsnon.copy()
        ratesum = S(0)
        crosssections(st, *key)
        if not st.rpexist[key]:
            continue
        N = st.Nenrp
        Egrid = st.Ein
        for nE in range(1, NintE + 1):
            E = Eint[nE]
            if E < Egrid[1] or E >= Egrid[N]:
                continue
            nen = locate(Egrid, 1, N, E)
            if nen == 0:
                continue
            xs = pol1(Egrid[nen], Egrid[nen + 1], st.xsrp[nen], st.xsrp[nen + 1], E)
            if k0 == 1:
                ratesum = ratesum + G[nE] * phi[nE] * xs
            else:
                ratesum = ratesum + phi[nE] * xs * dEint[nE]
        # Conversion of mb to cm^2: 1.e-27
        if k0 <= 1:
            st.reaction_rate[key] = st.fluxtotal * ratesum * S(1e-27)
            st.sacs[key] = ratesum
        else:
            st.reaction_rate[key] = projrate_density * ratesum * S(1e-27)
            if st.Leff > 0:
                st.sacs[key] = ratesum / st.Leff

    # Charged particles: output of stopping power
    if k0 > 1:
        reaction = "(" + st.ptype0 + ",x)"
        quantity = "Stopping power"
        with open(os.path.join(st.outdir, "stopping_power.txt"), "w", encoding="utf-8",
                  newline="\n") as fh:
            topline = st.targetnuclide.rstrip() + reaction + " " + quantity
            yandf.write_header(fh, 0, topline, st.source, st.user, st.date, st.oformat)
            yandf.write_target(fh, 0, st.Ztarget, st.Atarget, st.targetnuclide)
            yandf.write_reaction(fh, 0, reaction, 0.0, 0.0, 0, 0)
            yandf.write_quantity(fh, 2, quantity)
            yandf.write_datablock(fh, 2, 2, NintE, ["Energy", "Stopping power"], ["MeV", "MeV/cm"])
            for nE in range(1, NintE + 1):
                fh.write(es(Eint[nE], 15, 6) + es(S(1) / phi[nE], 15, 6) + "\n")
