"""Reactions on products (stage 3 of multiple hits): daughter rates and the network window.

Once the target is hit, its products can be hit too, and so can theirs. This
module finds every nuclide the beam can reach inside a user-set A/Z window and
computes its reaction rates, for the matrix solver (``irradiation.irradiate``).

How the rates are computed
--------------------------
ISOTOPIA computes every reaction rate as a weighted sum over an energy grid
(reactionrates.f90 lines 83-221 and 235-266; manual Eq. 4.11 and 4.20-4.22):

    R = scale * sum_k w_k sigma(E_k) * 1e-27        (sigma in mb, R in s^-1)

| Beam          | Grid E_k                       | Weight w_k                    | Scale      |
|---------------|--------------------------------|-------------------------------|------------|
| p d t h a     | 100 slices, E_back to E_beam   | dE / S(E_k), path length [cm] | n_p        |
| neutrons      | reactor spectrum groups        | spectrum x self-shielding G_k | total flux |
| photons       | bremsstrahlung spectrum        | spectrum x dE                 | total flux |

The weights depend only on the beam and the target material, never on the
reaction. ``rate_grid`` rebuilds them with the port's own functions, and
``check_rate_grid`` confirms that they reproduce every rate the port computed
for the target. A daughter's rates are then the same sum over its own library
files: no new stopping power, spectrum or rate code.

Physics assumptions (MIDAS Status Update, slide 40):
* Daughters sit in the same target and see the same beam: the same spectrum,
  the same stopping power (the target's), and for neutrons the same flux
  depression by the target's resonances (G_k). Their own concentration is
  far too small to shield anything.
* Rates are averaged over the target depth, as for the target in ISOTOPIA.
  A daughter is made mostly where its production cross section peaks, so its
  real second-hit rate can differ from the depth-averaged one. Not yet checked.
* Burnup of each hit nuclide follows what ISOTOPIA actually does for the
  target (team decision, 19 Nov 2026; see ``burnup_file``): the non-elastic
  cross section (MT005) for charged particles and photons, and for neutrons
  the total cross section (MT001), elastic scattering included, or MT003
  without self-shielding. ISOTOPIA's comments say non-elastic for neutrons
  too, but its file reread (reactionrates.f90 lines 233-236; crosssections.f90
  lines 86-89) leaves the total in place. Reactions that leave the nuclide
  unchanged (inelastic, and for neutrons also elastic, scattering) count as
  burnup but are not products, so atoms leave the bookkeeping while a nuclide
  is hit; a warning says how much.
* Rates are constant during irradiation; nothing is hit during cooling.

The network window
------------------
``Window`` sets how far the network may reach from the target, in A and Z,
below and above. Inside it, every nuclide that has cross sections in the
library is hit. Reaction products outside it go to one bin, ``OUTSIDE``
("outside_window"), so their atoms are counted; a warning gives each one's
rate. Decay is always followed, inside the window or not.

The library has unstable nuclides as targets only if they live about a day
or longer (shortest found: 8.75e4 s), plus some long-lived isomers. Other
nuclides in the window cannot be hit; a warning lists them.
"""
from __future__ import annotations

import copy
import os
import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from .constants import AVOGADRO, PARSYM, nuc
from .data import _read_xs_table
from .decaylib import OUTSIDE, DecayLibrary, Nuclide, label
from .fortran import S, locate, pol1, sexp
from .irradiation import Irradiation, irradiate, isotopia_times
from .matrix import Inventory, Reactions
from .rates import _charged_particle_spectrum, _read_neutron_spectrum, _read_photon_spectrum

# Tolerance of the check that the rebuilt weights reproduce the port's rates. The
# port sums in float32 (relative precision 6e-8 per term), the rebuild in float64.
GRID_RTOL = 1e-5
# Report product rates that differ from the burnup rate by more than this.
BALANCE_TOL = 1e-3

_STATE_LETTER = {"": None, "g": 0, "m": 1, "n": 2}


class RateGridError(RuntimeError):
    """The rebuilt weights do not reproduce the port's reaction rates."""


# ----------------------------------------------------------------------------
# The energy grid and its weights
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RateGrid:
    """ISOTOPIA's integration grid for one irradiation.

    Attributes
    ----------
    energies : numpy.ndarray
        Grid energies E_k [MeV].
    weights : numpy.ndarray
        w_k: path length dE/S(E_k) [cm] for charged particles; spectrum
        fraction times self-shielding (neutrons) or times dE (photons) [-].
    scale : float
        n_p, beam particles per volume per second [cm^-3 s^-1], for charged
        particles; total flux [cm^-2 s^-1] for neutrons and photons.
    """

    energies: np.ndarray
    weights: np.ndarray
    scale: float

    def rate(self, E, xs) -> float:
        """Reaction rate [s^-1] for a cross section given at energies E [MeV] in mb.

        Linear interpolation onto the grid, zero outside the tabulated range,
        as reactionrates.f90 does (lines 244-256).
        """
        E, xs = np.asarray(E, dtype=float), np.asarray(xs, dtype=float)
        if len(E) < 2:
            return 0.0
        inside = (self.energies >= E[0]) & (self.energies < E[-1])
        sigma = np.where(inside, np.interp(self.energies, E, xs), 0.0)
        return float(self.scale * np.dot(self.weights, sigma) * 1e-27)


def rate_grid(st) -> RateGrid:
    """Rebuild the grid and weights of ``rates._reactionrates`` after a port run.

    Mirrors reactionrates.f90 lines 83-221 (rates.py, ``_reactionrates``):
    the same spectra, stopping power, self-shielding and scale, from the
    State that run left behind.

    Parameters
    ----------
    st : State
        Port state after ``run`` (or after ``rates.reactionrates``).

    Returns
    -------
    RateGrid
        Energies [MeV], weights and scale (units in ``RateGrid``).
    """
    k0 = st.k0
    if k0 == 0:                                   # photons: normalised bremsstrahlung spectrum
        E, phi = _read_photon_spectrum(st)
        E, phi = np.array(E, dtype=float), np.array(phi, dtype=float)
        dE = np.empty_like(E)
        dE[0], dE[-1] = 0.5 * (E[1] - E[0]), 0.5 * (E[-1] - E[-2])
        dE[1:-1] = 0.5 * (E[2:] - E[:-2])
        norm = np.dot(phi, dE)
        weights = phi / norm * dE if norm > 0 else phi * dE
        return RateGrid(E, weights, float(st.fluxtotal))

    if k0 == 1:                                   # neutrons: reactor spectrum x self-shielding
        E, _, phi = _read_neutron_spectrum(st)
        G = np.ones(len(E))
        if st.flagselfshield:
            number_density = AVOGADRO / S(st.Atarget) * st.rho_target
            for k, e in enumerate(E):
                nen = locate(st.Etot, 1, st.Nentot, e)
                if nen == 0:
                    continue
                xst = pol1(st.Etot[nen], st.Etot[nen + 1], st.xstot[nen], st.xstot[nen + 1], e)
                term = number_density * xst * S(1e-27) * st.thickness
                if term > 0:
                    G[k] = (S(1) - sexp(-term)) / term
        return RateGrid(np.array(E, dtype=float), G * np.array(phi, dtype=float),
                        float(st.fluxtotal))

    # Charged particles: 100 slices from E_back to E_beam, weight = phi dE with phi = 1/S
    # (Bethe-Bloch at the midpoint, or the bin mean from the SRIM range table). Same helper
    # as the port, on a copy so the State the run left behind is not touched.
    _, E, dE, phi, projrate_density = _charged_particle_spectrum(copy.copy(st))
    weights = np.array(phi[1:], dtype=float) * np.array(dE[1:], dtype=float)
    return RateGrid(np.array(E[1:], dtype=float), weights, float(projrate_density))


# ----------------------------------------------------------------------------
# Library files of any nuclide as a target
# ----------------------------------------------------------------------------

def _name(nuclide: Nuclide) -> str:
    """Library name of a nuclide as a target, e.g. (42, 99, 0) -> 'Mo099', (47, 110, 1) -> 'Ag110m'."""
    z, a, i = nuclide
    return f"{nuc(z).strip()}{a:03d}" + {0: "", 1: "m"}.get(i, f"?{i}")


def tables_dir(st, nuclide: Nuclide) -> str:
    """Library folder of ``nuclide`` as a target for this projectile (checkfiles.f90 layout)."""
    return f"{st.crosspath.rstrip()}{PARSYM[st.k0]}/{_name(nuclide)}/{st.libname}/tables/"


def burnup_file(st, nuclide: Nuclide) -> str:
    """Cross-section file ISOTOPIA's R_T comes from, for ``nuclide`` as a target.

    The last file data.py ``crosssections(0, 0, -1)`` reads before the rate is
    summed (crosssections.f90 lines 82-89; reactionrates.f90 lines 233-236):
    MT001 (total) for neutrons with self-shielding, MT003 for neutrons
    without, MT005 (non-elastic) for every other projectile.
    """
    if st.k0 == 1:
        mt = "MT001" if st.flagselfshield else "MT003"
    else:
        mt = "MT005"
    return f"{tables_dir(st, nuclide)}xs/{PARSYM[st.k0]}-{_name(nuclide)}-{mt}.{st.libname}"


def _residual_files(st, nuclide: Nuclide) -> tuple[dict[Nuclide, str], dict[tuple, str]]:
    """Residual-production files of a target, one per product state.

    Products with isomers have files per state (g, m, n) besides the total;
    the states are used, as ``Irradiation.from_state`` does. Products without
    isomers have only the total, which is their ground state.

    Returns the files used per product state, and the total file of each
    product that is split into states (to check that the states add up).
    """
    folder = tables_dir(st, nuclide) + "residual/"
    pattern = re.compile(rf"{PARSYM[st.k0]}-{_name(nuclide)}-rp(\d{{3}})(\d{{3}})([gmn]?)"
                         rf"\.{re.escape(st.libname)}$")
    totals, states = {}, {}
    for f in sorted(os.listdir(folder)):
        m = pattern.match(f)
        if m:
            z, a, letter = int(m[1]), int(m[2]), _STATE_LETTER[m[3]]
            if letter is None:
                totals[(z, a, 0)] = folder + f
            else:
                states[(z, a, letter)] = folder + f
    with_states = {(z, a) for z, a, _ in states}
    files = {k: v for k, v in totals.items() if k[:2] not in with_states}
    files.update(states)
    split_totals = {k[:2]: v for k, v in totals.items() if k[:2] in with_states}
    return files, split_totals


def _channel_rates(st, grid: RateGrid, nuclide: Nuclide):
    """Burnup, product and own rates [s^-1] of a nuclide, and isomer-file checks; or None.

    Returns (burnup rate, product rates, rate back to the nuclide itself,
    {(Z, A): sum of the state rates / rate of the total file}).
    """
    path = burnup_file(st, nuclide)
    if not os.path.isfile(path):
        return None
    loss = grid.rate(*_read_xs_table(path))
    files, split_totals = _residual_files(st, nuclide)
    products, own, by_za = {}, 0.0, {}
    for product, path in files.items():
        rate = grid.rate(*_read_xs_table(path))
        by_za[product[:2]] = by_za.get(product[:2], 0.0) + rate
        if product == nuclide:
            own = rate
        elif rate > 0.0:
            products[product] = rate
    split = {}
    for za, path in split_totals.items():
        total = grid.rate(*_read_xs_table(path))
        if total > 0:
            split[za] = by_za.get(za, 0.0) / total
    return loss, products, own, split


def nuclide_reactions(st, grid: RateGrid, nuclide: Nuclide) -> Reactions | None:
    """Burnup and product rates of any nuclide in the target, from its library files.

    Parameters
    ----------
    st : State
        Port state after ``run``: projectile and library location.
    grid : RateGrid
        Weights of this irradiation (``rate_grid``).
    nuclide : Nuclide
        The nuclide hit, (Z, A, I).

    Returns
    -------
    Reactions or None
        Burnup rate (``burnup_file``) and product rates [s^-1]; reactions that
        leave the nuclide unchanged are not products. None if the library has
        no data for it as a target.
    """
    rates = _channel_rates(st, grid, nuclide)
    return None if rates is None else Reactions(rates[0], rates[1])


def check_rate_grid(st, grid: RateGrid) -> None:
    """Confirm that the rebuilt weights reproduce every target rate of the port run.

    Checks R_T and each residual rate in ``st.reaction_rate`` to ``GRID_RTOL``.
    R_T is recomputed from ``burnup_file``, the file the port's R_T comes from.

    Raises
    ------
    RateGridError
        Naming the first rate that differs: daughter rates would be wrong.
    """
    target = (st.Ztarget, st.Atarget, 0)
    folder = tables_dir(st, target)
    checks = {(0, 0, -1): burnup_file(st, target)}
    for (z, a, is_), r in st.reaction_rate.items():
        if (z, a) != (0, 0) and r > 0 and not st.xsfile.get((z, a, is_), "").strip():
            letter = {-1: "", 0: "g", 1: "m", 2: "n"}[is_]
            checks[(z, a, is_)] = (f"{folder}residual/{PARSYM[st.k0]}-{_name(target)}-rp"
                                   f"{z:03d}{a:03d}{letter}.{st.libname}")
    for key, path in checks.items():
        mine, port = grid.rate(*_read_xs_table(path)), float(st.reaction_rate[key])
        if abs(mine - port) > GRID_RTOL * abs(port):
            raise RateGridError(f"rebuilt weights give {mine:.7e} s^-1 for {key} ({path}); "
                                f"the port computed {port:.7e} s^-1")


# ----------------------------------------------------------------------------
# The network window and the hit network
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    """How far the hit network may reach from the target (inclusive, in mass and charge).

    Attributes
    ----------
    A_below, A_above : int
        Mass numbers from A_target - A_below to A_target + A_above.
    Z_below, Z_above : int
        Charges from Z_target - Z_below to Z_target + Z_above.
    """

    A_below: int
    A_above: int
    Z_below: int
    Z_above: int

    def __post_init__(self):
        if min(self.A_below, self.A_above, self.Z_below, self.Z_above) < 0:
            raise ValueError("window extents must be >= 0")

    def contains(self, target: Nuclide, n: Nuclide) -> bool:
        """True if ``n`` lies inside the window around ``target``."""
        dz, da = n[0] - target[0], n[1] - target[1]
        return -self.Z_below <= dz <= self.Z_above and -self.A_below <= da <= self.A_above


def _keep_inside(rx: Reactions, target: Nuclide, window: Window, lost: dict) -> Reactions:
    """Move products outside the window into the OUTSIDE bin, recording their rates."""
    products: dict[Nuclide, float] = {}
    for p, rate in rx.products.items():
        if window.contains(target, p):
            products[p] = rate
        else:
            products[OUTSIDE] = products.get(OUTSIDE, 0.0) + rate
            lost[p] = max(lost.get(p, 0.0), rate)
    return Reactions(rx.loss, products)


def hit_reactions(st, irr: Irradiation, window: Window,
                  library: DecayLibrary | None = None) -> tuple[dict[Nuclide, Reactions], list[str]]:
    """Reactions of every nuclide the beam can hit inside the window.

    Starting from the target, follows reaction products and decay daughters.
    Each nuclide inside the window with library data gets its reactions; its
    products are followed in turn, until nothing new appears.

    Parameters
    ----------
    st : State
        Port state after ``run`` (one target isotope).
    irr : Irradiation
        ``Irradiation.from_state(st)``: the target's own rates are kept as the
        port computed them.
    window : Window
        Network limit around the target.
    library : DecayLibrary, optional
        Decay data, to follow decay daughters. Default: the ISOTOPIA library.

    Returns
    -------
    (dict of Nuclide to Reactions, list of str)
        Reactions for ``irradiate(..., reactions=...)``, and warnings.

    Raises
    ------
    RateGridError
        If the rebuilt weights do not reproduce the port's rates.
    """
    library = library or DecayLibrary()
    grid = rate_grid(st)
    check_rate_grid(st, grid)
    target = irr.target
    lost: dict[Nuclide, float] = {}
    reactions = {target: _keep_inside(Reactions(irr.burnup, irr.rates), target, window, lost)}
    no_data, unbalanced, split_off = [], [], []

    # Breadth-first search over everything the target can turn into.
    queue = deque(reactions[target].products)
    seen = {target, OUTSIDE}
    while queue:
        n = queue.popleft()
        if n in seen:
            continue
        seen.add(n)
        d = library.get(n)
        queue.extend(m.daughter for m in (d.modes if d else ()) if m.daughter is not None)
        if not window.contains(target, n):
            continue
        rates = _channel_rates(st, grid, n)
        if rates is None:
            no_data.append(n)
            continue
        loss, products, own, split = rates
        split_off += [(n, za, f) for za, f in split.items() if abs(f - 1) > BALANCE_TOL]
        produced = sum(products.values())
        if loss > 0 and abs(produced / loss - 1) > BALANCE_TOL:
            unbalanced.append((n, produced / loss, own / loss))
        elif loss == 0 and produced > 0:
            unbalanced.append((n, np.inf, 0.0))
        reactions[n] = _keep_inside(Reactions(loss, products), target, window, lost)
        queue.extend(reactions[n].products)

    # Warnings: what is not hit, what leaves the window, what does not balance.
    warnings = []
    if no_data:
        warnings.append(f"Truncated chain: no cross sections in {st.crosspath.rstrip()}"
                        f"{PARSYM[st.k0]}/ for {len(no_data)} nuclide(s) in the window, which "
                        f"are therefore not hit: {', '.join(label(n) for n in sorted(no_data))}.")
    if lost:
        warnings.append("Truncated chain: products outside the window are collected in "
                        "'outside_window' (largest rate per product, s^-1): "
                        + ", ".join(f"{label(p)} {r:.2e}" for p, r in
                                    sorted(lost.items(), key=lambda x: -x[1])) + ".")
    if split_off:
        warnings.append("Isomer-state files that do not add up to the total file (the state "
                        "files are used, as for the target, so atoms are gained or lost): "
                        + ", ".join(f"{label(n)} -> {label((za[0], za[1], 0))} {f:.3g}"
                                    for n, za, f in split_off) + ".")
    if unbalanced:
        ratios = [r for _, r, _ in unbalanced]
        warnings.append(f"For {len(unbalanced)} hit nuclide(s) the product rates are "
                        f"{min(ratios):.3g} to {max(ratios):.3g} times the burnup rate "
                        "(as ISOTOPIA defines it for the target), so atoms are not conserved "
                        "while they are hit. Mostly reactions that leave the nuclide "
                        "unchanged (inelastic scattering; for neutrons also elastic), counted "
                        "in burnup but not as products; above 1, see the isomer-file warning. "
                        "Per nuclide, products / burnup and "
                        "inelastic-to-itself / burnup ('own'): "
                        + ", ".join(f"{label(n)} {r:.3g} (own {o:.3g})"
                                    for n, r, o in unbalanced) + ".")
    return reactions, warnings


def irradiate_with_hits(st, window: Window, times=None, library: DecayLibrary | None = None,
                        on_missing: str = "error") -> Inventory:
    """Irradiation and cooling with every nuclide in the window hit by the beam.

    Parameters
    ----------
    st : State
        Port state after ``run`` (one target isotope).
    window : Window
        Network limit around the target.
    times : iterable of float, optional
        Output times [s] from the start of irradiation. Default: ISOTOPIA's grid.
    library : DecayLibrary, optional
        Decay data. Default: the ISOTOPIA decay library.
    on_missing : {'error', 'warn'}
        Handling of nuclides absent from the decay library.

    Returns
    -------
    Inventory
        Atoms of every nuclide at each time, including 'outside_window', with
        all warnings.
    """
    library = library or DecayLibrary()
    irr = Irradiation.from_state(st)
    reactions, warnings = hit_reactions(st, irr, window, library)
    inv = irradiate(irr, isotopia_times(st) if times is None else times, library, on_missing,
                    reactions=reactions)
    inv.warnings.extend(warnings)
    return inv


__all__ = ["RateGrid", "RateGridError", "Window", "burnup_file", "check_rate_grid", "hit_reactions",
           "irradiate_with_hits", "nuclide_reactions", "rate_grid", "tables_dir"]
