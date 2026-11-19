"""Beam spectra, stopping power and reaction rates (stoppingpower.f90, reactionrates.f90)."""
from __future__ import annotations

import os

import numpy as np

from . import srim, yandf
from .constants import AMU, AVOGADRO, EMASS, PARMASS, PARSYM, PARZ, QELEM, nuc
from .data import crosssections
from .energyloss import RangeTable
from .fortran import S, es, i, list_reals, locate, pol1, sexp, slog, spow
from .state import IsotopiaError, State


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
    term1 = (S(0.307075) * st.rho_target * Z / st.molar_mass() * S(PARZ[k0] ** 2)
             / (beta * beta))
    term2 = S(0.5) * slog((Wmax * Wmax) / (Imean * Imean)) - beta * beta
    return term1 * term2


# ----------------------------------------------------------------------------
# Not in the Fortran: choice of stopping-power model, and thickness as an input
# ----------------------------------------------------------------------------

def _stop(msg: str):
    raise IsotopiaError(f" ISOTOPIA-error: {msg}")


def _bethe_range_table(st: State) -> RangeTable:
    """Bethe-Bloch (stoppingpower) as a path-length table, for solving Eback from a thickness.

    The table runs from the energy where the formula has its maximum up to 250 MeV. Below its
    maximum Bethe-Bloch turns over and then goes negative, so it is not used there.
    Energies in MeV, stopping power in MeV/cm.
    """
    grid = np.geomspace(1e-3, 250.0, 2001)
    with np.errstate(all="ignore"):
        Sp = np.array([float(stoppingpower(st, E)) for E in grid])
    first = int(np.argmax(np.where(np.isfinite(Sp), Sp, -np.inf)))
    return RangeTable(grid[first:], Sp[first:])


def setup_stopping(st: State) -> None:
    """Follow the beam through the stack of numbered layers, one of which is the target.

    Called once per isotope run for charged particles, after the target density is known.
    The beam crosses the layers in the order of their numbers ('layer <n> ...').

    1. Layers before the target: the beam loses energy in each. ``st.Ebeam_incident`` keeps
       the energy given as Ebeam; ``st.Ebeam`` becomes the mean energy entering the target,
       so everything downstream is unchanged.
    2. Target ('layer <n> target'): stopping-power model and, for a thickness input, Eback
       (``_setup_target``).
    3. Layers after the target: the beam leaving the target at Eback crosses them.

    Layers are pure elements and always use the SRIM tables (energyloss.RangeTable): density
    from the 'layer' line or constants.RHOTABLE (25 degC), atoms per cm3 from the CIAAW atomic
    weight the table was made with (srim_tables/index.csv). Each layer is uniform and the beam
    crosses it along its thickness; energies are mean energies (no straggling).

    A beam that stops before the target is an error (nothing reaches the target). A beam that
    stops in the target or in a later layer gives a warning, and the layers behind it are
    marked as not reached. Isotope production in the layers is calculated afterwards, one
    run per layer (run.layer_production).
    """
    warn = st.write if st.iso == 1 else (lambda line: None)   # once, not for every isotope
    particle = PARSYM[st.k0]
    ordered = sorted(st.layers, key=lambda layer: layer.number)
    front = [layer for layer in ordered if layer.number < st.target_layer]
    back = [layer for layer in ordered if layer.number > st.target_layer]

    # --- 1. Layers before the target ------------------------------------------------------
    st.Ebeam_incident = st.Ebeam
    E = float(st.Ebeam)
    for layer in front:
        n = layer.number
        table = _layer_range_table(particle, layer, f"layer {n}")
        layer.E_in, layer.E_out = E, float(table.exit_energy(E, layer.thickness))
        if layer.E_out == 0.0:
            _stop(f"the beam stops in layer {n} ({nuc(layer.Z).strip()}),"
                  f" {float(table.path_length(E)):.6E} cm into its {layer.thickness:.6E} cm;"
                  " no beam reaches the target")
        E = layer.E_out
    if front:
        st.Ebeam = S(E)
        if st.Eback == S(-1) and not st.thickness_given:
            st.Eback = max(st.Ebeam - S(5), S(0.1))     # the Fortran's default, from the target face
        if not st.thickness_given and st.Eback >= st.Ebeam:
            _stop(f"Eback must be below the energy at the target, {float(st.Ebeam):.6f} MeV"
                  " after the layers in front of it")

    # --- 2. Target ------------------------------------------------------------------------
    st.beam_stops = False
    _setup_target(st, warn)

    # --- 3. Layers after the target -------------------------------------------------------
    E = 0.0 if st.beam_stops else float(st.Eback)
    for layer in back:
        n = layer.number
        layer.reached = E > 0.0
        layer.E_in, layer.E_out, layer.reach = E, 0.0, 0.0
        if not layer.reached:
            continue
        table = _layer_range_table(particle, layer, f"layer {n}")
        if E < table.emin:
            _stop(f"Eback is below the SRIM stopping table ({table.emin * 1e3:g} keV),"
                  " so the layers behind the target cannot be calculated")
        layer.E_out = float(table.exit_energy(E, layer.thickness))
        if layer.E_out == 0.0:
            layer.reach = float(table.path_length(E))
            warn(f" ISOTOPIA-warning: the beam stops in layer {n} ({nuc(layer.Z).strip()}),"
                 f" {layer.reach:.6E} cm into its {layer.thickness:.6E} cm")
        E = layer.E_out


def _layer_range_table(particle: str, layer, name: str) -> RangeTable:
    """Path length versus energy [MeV -> cm] for one layer, from its SRIM table.

    Sets ``layer.mass`` [g/mol], the CIAAW atomic weight the table was made with.
    """
    try:
        table = srim.load_table(particle, layer.Z)
        layer.mass = float(srim.table_inputs()[layer.Z]["target_mass_u"])
        return RangeTable.from_srim(table, layer.rho, layer.mass)
    except (ValueError, OSError) as exc:
        _stop(f"{name}: {exc}")


def _setup_target(st: State, warn) -> None:
    """Choose the stopping-power model and, if 'thickness' is given, solve Eback from it.

    Sets
    ``st.stopping_model`` ("srim" or "bethe"), ``st.Amolar`` [g/mol], ``st.range_table``
    (path length versus energy for the target, energies in MeV and lengths in cm; None when
    Bethe-Bloch is used with Ebeam and Eback) and, for a thickness input, ``st.Eback`` [MeV].

    Model: keyword 'stopping'. Default: SRIM tables where they exist (p, d, t, h and a in
    the 92 elements H to U in isotopia_py/srim_tables), otherwise Bethe-Bloch with a warning. With
    ``st.fortran_defaults`` the default is Bethe-Bloch, as in the Fortran.

    Atoms per cm3 are N_A rho / M with M = ``st.molar_mass()``: the mass number of a
    single-isotope target (as in the Fortran) and the abundance-weighted mass number of a
    natural target, so that every isotope run of a natural target sees the same material,
    the same Eback and the same thickness. The Fortran's own configuration (Bethe-Bloch with
    Ebeam and Eback) keeps the mass number of each isotope, to reproduce its output.

    Thickness (manual Eq. 4.23 inverted, energyloss.RangeTable.exit_energy): Eback is the
    mean energy leaving a target of ``st.thickness`` cm. If the beam stops in the target,
    SRIM runs use the lowest table energy as Eback, write a warning, and count only the
    irradiated depth as the effective target; Bethe-Bloch runs stop with an error, because
    the formula is not valid there.
    """
    particle, element = PARSYM[st.k0], nuc(st.Ztarget).strip()

    # --- Model: the keyword, or SRIM where a table exists -------------------------------
    has_table = particle in srim.PARTICLES and (
        srim.TABLE_DIR / particle / f"{particle}_in_{element}.txt").exists()
    model = st.stopping or ("bethe" if st.fortran_defaults else "srim")
    if model == "srim" and not has_table:
        if st.stopping == "srim":
            _stop(f"no SRIM stopping table for {particle} in {element}; tables exist for p, d, t,"
                  " h and a in every element from H to U (Z = 1-92). Use stopping bethe")
        warn(f" ISOTOPIA-warning: no SRIM stopping table for {particle} in {element};"
             " the Bethe-Bloch formula is used instead")
        model = "bethe"
    st.stopping_model = model

    # --- Molar mass for atoms per cm3 ----------------------------------------------------
    fortran_case = model == "bethe" and not st.thickness_given
    if st.flagnatural and not fortran_case:
        st.Amolar = sum((st.abun[k] * S(st.isotope[k]) for k in range(1, st.isonum + 1)), S(0))
    else:
        st.Amolar = S(st.Atarget)

    # --- Path length versus energy for this target ---------------------------------------
    st.range_table = None
    try:
        if model == "srim":
            st.range_table = RangeTable.from_srim(
                srim.load_table(particle, st.Ztarget), float(st.rho_target), float(st.Amolar))
        table = st.range_table or (_bethe_range_table(st) if st.thickness_given else None)
        if table is not None and not table.emin <= float(st.Ebeam) <= table.emax:
            _stop(f"Ebeam is outside the stopping-power table ({table.emin:g} to {table.emax:g} MeV)")
    except (ValueError, OSError) as exc:
        _stop(str(exc))

    # --- Eback given: it must lie inside the table ---------------------------------------
    if not st.thickness_given:
        if model == "srim" and st.Eback < S(table.emin):
            _stop(f"Eback is below the SRIM stopping table ({table.emin * 1e3:g} keV)")
        return

    # --- Thickness given: solve Eback -----------------------------------------------------
    E_out = float(table.exit_energy(float(st.Ebeam), float(st.thickness)))
    if E_out == 0.0:
        reach = float(table.path_length(float(st.Ebeam)))
        if model == "bethe":
            _stop(f"the beam slows below {table.emin:.3g} MeV within {reach:.6E} cm of the target,"
                  " where the Bethe-Bloch formula is not valid; SRIM stopping tables are needed"
                  " for a target this thick")
        warn(f" ISOTOPIA-warning: the beam stops in the target, {reach:.6E} cm into its"
             f" {float(st.thickness):.6E} cm.")
        warn("   Only the irradiated depth counts as the effective target (thickness, volume, mass);")
        warn(f"   Eback is set to the lowest energy of the stopping table, {table.emin * 1e3:g} keV.")
        E_out = table.emin
        st.beam_stops = True
    st.Eback = S(E_out)
    if st.Eback >= st.Ebeam:
        _stop("the target is too thin: the energy lost in it is below single precision")


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


def _charged_particle_spectrum(st: State):
    """Energy grid and spectrum 1/S of a charged-particle beam slowing from Ebeam to Eback.

    Port of reactionrates.f90 lines 158-174: 100 equal energy bins between Eback and Ebeam,
    phi = 1/S [cm/MeV] in each, Leff = sum(phi dE) [cm] (manual Eqs. 4.23-4.25). Sets
    st.Leff [cm], st.V_target [cm3], st.heat [kW], st.projnum [1/s] and st.Eaverage [MeV].

    With Bethe-Bloch, phi is 1/S at the bin midpoint, as in the Fortran. With SRIM tables,
    phi is the mean of 1/S over the bin, from the exact path length of the bin
    (energyloss.RangeTable.thickness), so Leff equals the path length from Ebeam to Eback
    and, for a thickness input, the thickness that was given.

    Returns
    -------
    NintE : int
        Number of energy bins.
    Eint, dEint, phi : list of numpy.float32
        Bin midpoints [MeV], bin widths [MeV] and spectrum [cm/MeV]; index 0 is unused.
    projrate_density : numpy.float32
        Projectiles per cm3 of target per second.
    """
    k0 = st.k0
    NintE = 100
    dE = (st.Ebeam - st.Eback) / S(NintE)
    phisum = S(0)
    st.Eaverage = S(0)
    Eint = [S(0)] * (NintE + 1)
    dEint = [S(0)] * (NintE + 1)
    phi = [S(0)] * (NintE + 1)
    table = st.range_table if st.stopping_model == "srim" else None
    for nE in range(1, NintE + 1):
        Eint[nE] = st.Eback + (S(nE) - S(0.5)) * dE
        dEint[nE] = dE
        if table is None:
            Sp = stoppingpower(st, Eint[nE])
            if Sp != 0:
                phi[nE] = S(1) / Sp
        else:
            # Bin edges, kept inside the table (single-precision Eback can round just below it)
            E_lo = max(float(st.Eback) + (nE - 1) * float(dE), table.emin)
            E_hi = min(float(st.Eback) + nE * float(dE), table.emax)
            phi[nE] = S(float(table.thickness(E_hi, E_lo)) / float(dE))
        phisum = phisum + phi[nE] * dE
        st.Eaverage = st.Eaverage + Eint[nE] * phi[nE] * dEint[nE]
    st.Leff = phisum
    st.V_target = st.Area * st.Leff
    # Ibeam is in A, energies are in MeV, heat is in kW
    st.heat = S(1000) * st.Ibeam * (st.Ebeam - st.Eback) / S(PARZ[k0])
    st.projnum = st.Ibeam / (S(PARZ[k0]) * QELEM)
    projrate_density = st.projnum / st.V_target
    st.Eaverage = st.Eaverage / st.Leff
    return NintE, Eint, dEint, phi, projrate_density


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
        NintE, Eint, dEint, phi, projrate_density = _charged_particle_spectrum(st)
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
