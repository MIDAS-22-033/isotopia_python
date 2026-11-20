"""Output: the summary and per-nuclide .act files (prodout.f90) and the merged
files for natural targets (the second half of natural.f90).
"""
from __future__ import annotations

import os

from . import yandf
from .constants import HOURSEC, ISOCHAR, NUMISOM, NUMTIME, PARZ, nuc
from .fortran import S, a, es, f, i, time_string
from .state import State

_DATA_COLS = ["Time", "Activity", "Spec_activity", "Prod_rate", "#Isotopes", "Isot_fraction", "Time"]


def _yield_factor(st: State):
    if st.k0 > 1:
        return st.rfac * st.cfac * st.tfac
    return st.rfac / st.mfac * st.tfac


def _yield_string(st: State) -> str:
    if st.k0 > 1:
        return st.rstr + "/(" + st.cstr + "." + st.tstr + ")"
    return st.rstr + "/(" + st.ystr + "." + st.tstr + ")"


def _units(st: State) -> list[str]:
    return ["s", st.rstr, st.rstr + "/" + st.ystr, _yield_string(st), "", "", "h"]


def _halflife_string(st: State, key) -> str:
    if st.Thalf[key] > 1e17:
        return "stable"
    return time_string(st.Thalf[key], st.Td[key][1:])


def _maxprod_string(st: State, key) -> str:
    if st.Tmax[key] > 1e17:
        return "infinity"
    return time_string(st.Tmax[key], st.Tp[key][1:])


def _act_parameters(st: State, fh, key, N0, rate, lam, init_rate, act_eoi, spec_eoi,
                    with_sacs: bool) -> None:
    """The 'parameters' block shared by per-isotope and natural .act files."""
    id4 = 4
    if with_sacs:
        # prodout.f90
        if st.k0 != 1:
            yandf.write_real(fh, id4, "E-Beam [MeV]", st.Ebeam)
            if st.k0 > 1:
                yandf.write_real(fh, id4, "E-Back [MeV]", st.Eback)
            yandf.write_real(fh, id4, "Beam current [" + st.cstr + "]", st.Ibeam_input)
    else:
        # natural.f90
        if st.k0 > 1:
            yandf.write_real(fh, id4, "E-Beam [MeV]", st.Ebeam)
            yandf.write_real(fh, id4, "E-Back [MeV]", st.Eback)
        if st.k0 != 1:
            yandf.write_real(fh, id4, "Beam current [" + st.cstr + "]", st.Ibeam_input)
    yandf.write_real(fh, id4, "Number of target atoms", N0)
    yandf.write_real(fh, id4, "Reaction constant [s^-1]", rate)
    yandf.write_real(fh, id4, "Decay constant [s^-1]", lam)
    if with_sacs:
        yandf.write_real(fh, id4, "Spectrum averaged cross section [mb]", st.sacs[key])
        if st.k0 == 1:
            yandf.write_real(fh, id4, "Average self-shielding factor", st.selfshield_av)
    yandf.write_real(fh, id4, "Initial production rate [" + _yield_string(st) + "]", init_rate)
    yandf.write_real(fh, id4, "Total activity at EOI [" + st.rstr + "]", act_eoi)
    yandf.write_real(fh, id4, "Specific activity at EOI [" + st.rstr + "/" + st.ystr + "]", spec_eoi)
    yandf.write_char(fh, id4, "Irradiation time [s]", time_string(st.Tir, st.Tirrad[1:]))
    yandf.write_char(fh, id4, "Cooling time [s]", time_string(st.Tco, st.Tcool[1:]))
    yandf.write_char(fh, id4, "Half-life [s]", _halflife_string(st, key))
    yandf.write_char(fh, id4, "Maximum production [s]", _maxprod_string(st, key))


def _write_act(st: State, filename: str, key, finalnuclide: str, write_params, rows) -> None:
    iz, ia, _ = key
    quantity = "Isotope production"
    reaction = "(" + st.ptype0 + ",x)"
    with open(os.path.join(st.outdir, filename), "w", encoding="utf-8", newline="\n") as fh:
        topline = st.targetnuclide.rstrip() + reaction + finalnuclide.rstrip() + " " + quantity
        yandf.write_header(fh, 0, topline, st.source, st.user, st.date, st.oformat)
        yandf.write_target(fh, 0, st.Ztarget, st.Atarget, st.targetnuclide)
        yandf.write_reaction(fh, 0, reaction, 0.0, 0.0, 0, 0)
        yandf.write_residual(fh, 2, iz, ia, finalnuclide)
        yandf.write_char(fh, 2, "parameters", "")
        write_params(fh)
        yandf.write_quantity(fh, 2, quantity)
        yandf.write_datablock(fh, 2, 7, NUMTIME, _DATA_COLS, _units(st))
        for row in rows:
            fh.write("".join(es(v, 15, 6) for v in row) + "\n")


def _layers_out(st: State) -> None:
    """Table of the stack along the beam, one row per numbered layer (not in the Fortran).

    Energies are mean energies in MeV. Heat is the beam power left in the layer in kW,
    1000 * Ibeam [A] * (E-in - E-out) [MeV] / projectile charge number, as for the target
    (reactionrates.f90 line 171).
    """
    w = st.write
    kw_per_mev = S(1000) * st.Ibeam / S(PARZ[st.k0])
    w()
    w(" Layers along the beam (mean energies; layers use the SRIM stopping tables)")
    w()
    w(" Layer  Material  Thickness [cm]  Density [g/cm^3]     E-in [MeV]    E-out [MeV]      Heat [kW]")

    def row(n, material, thickness, rho, E_in, E_out, heat, note=""):
        w(i(n, 6) + "  " + a(material, 8) + es(thickness, 16, 6) + es(rho, 18, 6)
          + es(E_in, 15, 6) + es(E_out, 15, 6) + es(heat, 15, 6) + note)

    layers = {layer.number: layer for layer in st.layers}
    for n in range(1, len(layers) + 2):
        if n == st.target_layer:
            # The target row: effective thickness, and E-out 0 if the beam stops in it
            row(n, st.targetnuclide.strip(), st.Leff, st.rho_target, st.Ebeam,
                S(0) if st.beam_stops else st.Eback, st.heat,
                "   target; beam stops in it" if st.beam_stops else "   target")
            continue
        layer = layers[n]
        if not layer.reached:
            w(i(n, 6) + "  " + a(nuc(layer.Z).strip(), 8)
              + es(layer.thickness, 16, 6) + es(layer.rho, 18, 6) + "   not reached by the beam")
        else:
            note = f"   beam stops at {layer.reach:.6E} cm" if layer.E_out == 0.0 else ""
            row(n, nuc(layer.Z).strip(), layer.thickness, layer.rho, layer.E_in,
                layer.E_out, kw_per_mev * S(layer.E_in - layer.E_out), note)


def prodout(st: State) -> None:
    w = st.write
    yfac = _yield_factor(st)
    w()
    w(" Summary of isotope production for " + a(st.ptype0, 1) + " + " + st.targetnuclide.rstrip())
    w()
    w(" Maximal irradiation time [s]: " + time_string(st.Tir, st.Tirrad[1:]))
    w(" Cooling time [s]: " + time_string(st.Tco, st.Tcool[1:]))
    if st.k0 != 1:
        if st.layers:    # not in the Fortran: Ebeam was given before the front layers
            w(" Beam energy before the layers [MeV]:" + es(st.Ebeam_incident, 15, 6))
        w(" E-beam [MeV]:" + es(st.Ebeam, 15, 6))
        if st.k0 > 1:
            w(" E-back [MeV]:" + es(st.Eback, 15, 6))
    if st.k0 != 1:
        w(" Beam current [" + st.cstr + "]: " + f(st.Ibeam_input, 12, 3))
    w(" Target material density [g/cm^3]:" + es(st.rho_target, 15, 6))
    w(" Target area [cm^2]:" + es(st.Area, 15, 6))
    # Not in the Fortran: the stopping-power model and, if it was given, the thickness
    if st.k0 > 1 and not st.fortran_defaults:
        model = {"srim": "SRIM table " + a(st.ptype0, 1) + "_in_" + nuc(st.Ztarget).strip() + ".txt",
                 "bethe": "Bethe-Bloch formula"}[st.stopping_model]
        w(" Stopping power: " + model)
    if st.k0 > 1 and st.thickness_given:
        w(" Target thickness given [cm]:" + es(st.thickness, 15, 6) + "  (E-back calculated from it)")
    if st.k0 > 1:
        w(" Effective target thickness [cm]:" + es(st.Leff, 15, 6))
    else:
        w(" Target thickness [cm]:" + es(st.thickness, 15, 6))
    w(" Effective target volume [cm^3]:" + es(st.V_target, 15, 6))
    w(" Effective target mass [" + st.ystr + "]:" + es(st.M_target * st.mfac, 15, 6))
    w(" Average energy of spectrum [MeV]: " + es(st.Eaverage, 15, 6))
    w(" Number of target atoms: " + es(st.N_0, 15, 6))
    if st.k0 > 1:
        w(" Number of incident particles [s^-1]:" + es(st.projnum, 15, 6))
        w(" Produced heat in target [kW]:" + es(st.heat, 15, 6))
        if st.layers:
            _layers_out(st)
    else:
        w(" Total flux [cm^-2s^-1]:" + es(st.fluxtotal, 15, 6))
    w()
    w(" (Maximum) production and decay rates per isotope")
    w()
    w(" Total production rate [s^-1]:" + es(st.reaction_rate[(0, 0, -1)], 15, 6))
    w()
    w("#  Nuc       Activity     Spec_activity    Prod_rate      #Isotopes Isot_fraction "
      " Reac_constant   Decay_constant            Half-life                      Time of maximum production")
    w("#              [" + st.rstr + "]          [" + st.rstr + "/" + st.ystr + "]        ["
      + _yield_string(st) + "]        []            []" + "        [s^-1]         [s^-1]       ")
    keys = _product_keys(st)
    for key in keys:
        iz, ia, is_ = key
        it = int(st.Tmaxactivity[key])
        if it == 0:
            it = st.Ntime
        if st.Thalf[key] > 1e17:
            halflife = a("       stable  ", 38)
        else:
            Td = st.Td[key]
            halflife = (i(Td[1], 11) + " y " + i(Td[2], 3) + " d " + i(Td[3], 3) + " h "
                        + i(Td[4], 3) + " m " + i(Td[5], 3) + " s ")
        if st.Tmax[key] > 1e17:
            maxprod = a("       infinite", 38)
        else:
            Tp = st.Tp[key]
            maxprod = (i(Tp[1], 11) + " y " + i(Tp[2], 3) + " d " + i(Tp[3], 3) + " h "
                       + i(Tp[4], 3) + " m " + i(Tp[5], 3) + " s ")
        if is_ == -1 and st.Niso[(iz, ia, 0)][it] > 0 and st.Niso[(iz, ia, 1)][it] > 0:
            halflife = " " * 38
            maxprod = " " * 38
        w(" " + a(nuc(iz), 2) + i(ia, 4) + " " + ISOCHAR[is_]
          + es(st.activity[key][it] * st.rfac, 15, 6)
          + es(st.specactivity[key][it] * st.rfac / st.mfac, 15, 6)
          + es(st.yld[key][1] * yfac, 15, 6)
          + es(st.Niso[key][it], 15, 6)
          + f(st.Nisorel[key][it], 10, 5)
          + es(st.reaction_rate[key], 15, 6)
          + es(st.lam[key], 15, 6)
          + a(halflife, 38) + a(maxprod, 38))

    # Output of files per residual product
    w()
    for key in keys:
        iz, ia, is_ = key
        if st.flagZAoutput:
            name = "Y" + i(iz, 3, 3) + i(ia, 3, 3) + ".tot"
            if is_ >= 0:
                name = name[:8] + "L00" + name[11:]
            if is_ >= 1:
                name = name[:9] + i(is_, 2, 2) + name[11:]
            Yfile = name + st.natstring[st.iso]
        else:
            Yfile = nuc(iz).strip() + i(ia, 3, 3) + {0: "g", 1: "m", 2: "n"}.get(is_, "")
            Yfile = Yfile + ".act" + st.natstring[st.iso]
        Yfile = Yfile.rstrip()[:15]
        finalnuclide = nuc(iz).strip() + str(ia) + ISOCHAR[is_]
        w("file: " + Yfile)

        def params(fh, key=key):
            _act_parameters(st, fh, key, st.N_0, st.reaction_rate[key], st.lam[key],
                            st.yld[key][1] * yfac, st.activity[key][st.Ntime] * st.rfac,
                            st.specactivity[key][st.Ntime] * st.rfac / st.mfac, with_sacs=True)

        rows = [(st.Tgrid[it], st.activity[key][it] * st.rfac,
                 st.specactivity[key][it] * st.rfac / st.mfac, st.yld[key][it] * yfac,
                 st.Niso[key][it], st.Nisorel[key][it], st.Tgrid[it] / HOURSEC)
                for it in range(1, NUMTIME + 1)]
        _write_act(st, Yfile, key, finalnuclide, params, rows)
    w()
    w("  End of ISOTOPIA calculation for " + a(st.ptype0, 1) + " + " + st.targetnuclide.rstrip())


def _product_keys(st: State) -> list[tuple[int, int, int]]:
    keys = []
    for iz in range(st.Zcomp + 1, st.Zcomp + 1 - st.Zdepth - 1, -1):
        for ia in range(st.Acomp, st.Acomp - st.Adepth - 1, -1):
            for is_ in range(-1, st.Nisomer[(iz, ia)] + 1):
                if st.Yexist[(iz, ia, is_)]:
                    keys.append((iz, ia, is_))
    return keys


def natural_output(st: State) -> None:
    """Merge the per-isotope results of a natural target (natural.f90)."""
    yfac = _yield_factor(st)
    st.write()
    st.targetnuclide = st.Starget.strip() + "0"
    for iz in range(st.Zcomp + 1, st.Zcomp + 1 - st.Zdepth - 1, -1):
        for ia in range(st.Acomp, st.Acomp - st.Adepth - 1, -1):
            for is_ in range(-1, NUMISOM + 1):
                key = (iz, ia, is_)
                if not st.Ynatexist[key]:
                    continue
                if st.flagZAoutput:
                    Yfile0 = "Y" + i(iz, 3, 3) + i(ia, 3, 3) + ".tot"
                    if is_ in (0, 1, 2):
                        Yfile0 = Yfile0[:8] + "L0" + str(is_)
                else:
                    Yfile0 = (nuc(iz).strip() + i(ia, 3, 3) + {0: "g", 1: "m", 2: "n"}.get(is_, "")
                              + ".act")
                # adjustl without trim: masses below 100 keep a trailing blank
                finalnuclide = nuc(iz).strip() + str(ia).ljust(3) + ISOCHAR[is_]
                st.write("file: " + Yfile0)

                def params(fh, key=key):
                    _act_parameters(st, fh, key, st.N_0_nat, st.reaction_ratenat[key], st.lam[key],
                                    st.yieldnat[key][1] * yfac,
                                    st.activitynat[key][st.Ntime] * st.rfac,
                                    st.specactivitynat[key][st.Ntime] * st.rfac / st.mfac,
                                    with_sacs=False)

                rows = [(st.Tgrid[it], st.activitynat[key][it] * st.rfac,
                         st.specactivitynat[key][it] * st.rfac / st.mfac,
                         st.yieldnat[key][it] * yfac, st.Nisonat[key][it],
                         st.Nisorelnat[key][it], st.Tgrid[it] / HOURSEC)
                        for it in range(1, NUMTIME + 1)]
                _write_act(st, Yfile0, key, finalnuclide, params, rows)
    zs = list(range(st.Zcomp + 1, st.Zcomp + 1 - st.Zdepth - 1, -1))
    with open(os.path.join(st.outdir, "elements.out"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Time [h]   " + "".join(a(nuc(iz), 2) + "      " for iz in zs) + "\n")
        for it in range(1, st.Ntime + 1):
            fh.write(f(st.Tgrid[it], 8, 1) + "".join(f(st.Nelrel[iz][it], 8, 5) for iz in zs) + "\n")
    st.write()
    st.write("  End of ISOTOPIA for natural target")


def layer_products(sub: State) -> list[tuple[tuple[int, int, int], float, float]]:
    """Radioactive products of a finished run with their activity at the end of irradiation.

    Parameters
    ----------
    sub : State
        A finished run (single isotope or natural element).

    Returns
    -------
    list of ((Z, A, I), activity, half-life)
        Activity in Bq at the end of irradiation, half-life in s, most active first. I is -1
        for a nuclide without separate isomer results, else 0 (ground state), 1 or 2.
    """
    natural = sub.isonum > 1
    exist = sub.Ynatexist if natural else sub.Yexist
    activity = sub.activitynat if natural else sub.activity
    keys = [key for key, there in exist.items() if there]
    products = []
    for iz, ia, is_ in keys:
        # the sum over isomers (-1) is skipped when the isomers are listed themselves
        if is_ == -1 and any((iz, ia, m) in keys for m in (0, 1, 2)):
            continue
        act = float(activity[(iz, ia, is_)][sub.Ntime])
        if act > 0:
            products.append(((iz, ia, is_), act, float(sub.Thalf[(iz, ia, is_)])))
    return sorted(products, key=lambda p: -p[1])


def layer_production_out(st: State, results) -> None:
    """Summary of the isotope production in the layers (not in the Fortran).

    Parameters
    ----------
    st : State
        The target run; the summary goes to its output.
    results : list of (Layer, folder, State or None, message)
        One entry per layer the beam reaches, from run.layer_production: the layer, the name
        of its output folder, its finished run (None if it failed) and the error message.
    """
    w = st.write
    w()
    w(" Isotope production in the layers")
    w(" Each layer is run as a target of its natural element; full output in the folder given.")
    w(" Activities at the end of irradiation; only the primary beam is followed.")
    for layer, folder, sub, message in results:
        element = nuc(layer.Z).strip()
        w()
        w(f" Layer {layer.number} ({element}), {layer.E_in:.6f} -> {layer.E_out:.6f} MeV,"
          f" folder {folder}")
        if sub is None:
            w(f" ISOTOPIA-warning: no production calculated for layer {layer.number} ({element}):")
            w("   " + message.replace("ISOTOPIA-error: ", ""))
            continue
        products = layer_products(sub)
        if not products:
            w("   no radioactive products")
            continue
        w("   Nuclide   Activity [" + st.rstr + "]   Half-life [s]")
        for (iz, ia, is_), act, thalf in products:
            w("   " + a(nuc(iz), 2) + i(ia, 4) + " " + ISOCHAR[is_]
              + es(S(act) * st.rfac, 15, 6) + es(thalf, 15, 6))
    for layer in st.layers:
        if not layer.reached:
            w()
            w(f" Layer {layer.number} ({nuc(layer.Z).strip()}): not reached by the beam,"
              " no production")
