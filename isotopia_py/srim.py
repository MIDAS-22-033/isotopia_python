"""SRIM stopping tables for ISOTOPIA projectiles in the natural elements (new; not in the Fortran).

The tables in ``srim_tables/<particle>/<particle>_in_<Symbol>.txt`` (e.g. ``srim_tables/p/p_in_Ni.txt``)
are SRIM SR Module output for protons (``p``), deuterons (``d``), tritons (``t``), helium-3
(``h``) and alphas (``a``) in the 84 elements that have a
CIAAW standard atomic weight and in the radioactive elements up to U (all 92 from H to U),
from 1 keV to 250 MeV. ``srim_tables/README.md`` gives how they were
made, and ``srim_tables/index.csv`` lists the density and atomic mass each one was made with.

The tables give stopping *per atom*, epsilon(E), in eV/(1E15 atoms/cm2). It depends on the
target element but not on its density. This module converts it to a stopping power per unit
length for the density and atomic mass of the actual target:

    S(E) [MeV/cm] = epsilon(E) [eV/(1E15 atoms/cm2)] * 1e-21 * n,    n = N_A * rho / M

where n is the number of atoms per cm3. The factor 1e-21 is 1e-15 (atoms/cm2 in the unit) times
1e-6 (eV -> MeV). This is the same conversion SRIM uses for the factors at the bottom of each table.

Assumptions and limits
- Electronic stopping depends on the target element (Z), not on its isotope, so one table per
  element serves natural and enriched targets; the isotope enters only through n = N_A rho / M.
- Nuclear stopping depends on the target atomic mass (roughly as 1/(M_ion + M)). The tables use
  the natural (CIAAW) mass, so for an enriched target the nuclear part is off by about the
  relative mass difference (e.g. 4% for Mo-100). The nuclear share of the total is at most 3%
  for protons and 24% for alphas above 10 keV, and at most 2.3% for both above 100 keV (all
  84 tables), so this matters only near the end of the range.
- Between table energies the stopping is interpolated linearly in log(E) and log(epsilon).
- Energies outside the table (below ~1 keV or above 250 MeV) raise ValueError; nothing is
  extrapolated.
- Used by rates.setup_stopping: SRIM is the default stopping power of a charged-particle run
  where a table exists (keyword 'stopping'). The Fortran's Bethe-Bloch formula stays available.

Reference: J. F. Ziegler, M. D. Ziegler and J. P. Biersack, "SRIM - The stopping and range of
ions in matter (2010)", Nucl. Instrum. Methods B 268, 1818 (2010), doi:10.1016/j.nimb.2010.02.091.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import NUC, PARMASS, PARSYM, PARZ

# Shipped tables (licence terms in srim_tables/NOTICE) and the ISOTOPIA projectiles they cover
TABLE_DIR = Path(__file__).with_name("srim_tables")
PARTICLES = ("p", "d", "t", "h", "a")

# Avogadro constant, mol^-1 (exact, SI 2019), in double precision. constants.AVOGADRO is the
# Fortran's single-precision copy, kept there for digit-identical sample output.
N_A = 6.02214076e23

# SRIM energy units -> MeV
_E_UNIT = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}


# ---------------------------------------------------------------------------------------------
# One table
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoppingTable:
    """SRIM stopping per atom for one projectile in one element.

    Attributes
    ----------
    particle : str
        ISOTOPIA projectile symbol: 'p', 'd', 't', 'h' (helium-3) or 'a'.
    Z : int
        Atomic number of the target element.
    symbol : str
        Element symbol of the target.
    energy : numpy.ndarray
        Projectile kinetic energies of the table rows, MeV (increasing).
    electronic, nuclear : numpy.ndarray
        Electronic and nuclear stopping per atom at ``energy``, eV/(1E15 atoms/cm2).
    gas : bool
        True if SRIM treated the target as a gas.
    srim_density : float
        Density SRIM was given, g/cm3 (from the file header).
    srim_atom_density : float
        Atoms per cm3 SRIM used, atoms/cm3 (from the file header; = N_A * density / mass).
    path : pathlib.Path
        File the table was read from.
    """

    particle: str
    Z: int
    symbol: str
    energy: np.ndarray
    electronic: np.ndarray
    nuclear: np.ndarray
    gas: bool
    srim_density: float
    srim_atom_density: float
    path: Path

    @property
    def emin(self) -> float:
        """Lowest table energy, MeV."""
        return float(self.energy[0])

    @property
    def emax(self) -> float:
        """Highest table energy, MeV."""
        return float(self.energy[-1])

    def per_atom(self, E, component: str = "total") -> np.ndarray:
        """Stopping per atom at energy E, interpolated log-log between table rows.

        Parameters
        ----------
        E : float or array_like
            Projectile kinetic energy, MeV. Must lie within [emin, emax].
        component : {'total', 'electronic', 'nuclear'}
            Which part of the stopping to return.

        Returns
        -------
        numpy.ndarray
            Stopping per atom, eV/(1E15 atoms/cm2), same shape as E.
        """
        E = np.asarray(E, dtype=float)
        if np.any(E < self.emin) or np.any(E > self.emax):
            raise ValueError(
                f"{self.particle} in {self.symbol}: energy outside the SRIM table "
                f"({self.emin:g} to {self.emax:g} MeV); no extrapolation is done"
            )
        if component == "total":
            values = self.electronic + self.nuclear
        elif component in ("electronic", "nuclear"):
            values = getattr(self, component)
        else:
            raise ValueError(
                f"component must be 'total', 'electronic' or 'nuclear', not {component!r}"
            )
        # Linear interpolation in log-log space: on a log grid, stopping varies smoothly as a power law.
        return np.exp(np.interp(np.log(E), np.log(self.energy), np.log(values)))

    def stopping_power(
        self, E, rho: float, mass: float, component: str = "total"
    ) -> np.ndarray:
        """Stopping power per unit length for a target of given density and atomic mass.

        S = epsilon * 1e-21 * N_A * rho / M (see module docstring).

        Parameters
        ----------
        E : float or array_like
            Projectile kinetic energy, MeV.
        rho : float
            Target density, g/cm3.
        mass : float
            Atomic mass of the target material, g/mol (u): the standard atomic weight for a
            natural element, the isotope mass for an enriched target.
        component : {'total', 'electronic', 'nuclear'}
            Which part of the stopping to return.

        Returns
        -------
        numpy.ndarray
            dE/dx, MeV/cm, same shape as E.
        """
        if rho <= 0 or mass <= 0:
            raise ValueError("rho and mass must be positive")
        atoms_per_cm3 = N_A * rho / mass
        return self.per_atom(E, component) * 1e-21 * atoms_per_cm3


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

# Header and row patterns. The density line reads "Density = ..." in SR Module output and
# "Target Density = ..." in output from SRIM's stopping-table window; both match.
_ION = re.compile(r"Ion = \w+\s*\[(\d+)\]\s*, Mass = ([\d.]+) amu")
_DENSITY = re.compile(r"Density =\s*(\S+) g/cm3 =\s*(\S+) atoms/cm3")
_UNITS = re.compile(r"Stopping Units =\s*(.+)")
_TARGET = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", re.MULTILINE
)
_ROW = re.compile(
    r"^\s*([\d.]+)\s*(eV|keV|MeV|GeV)\s+(\d\.\d+E[+-]\d+)\s+(\d\.\d+E[+-]\d+)\s",
    re.MULTILINE,
)


def read_table(path: str | Path, particle: str) -> StoppingTable:
    """Read one SRIM stopping table (SR Module or SRIM stopping-table window output).

    Parameters
    ----------
    path : str or pathlib.Path
        SRIM output file. Its stopping units must be eV/(1E15 atoms/cm2), and the target
        must be a single element.
    particle : str
        ISOTOPIA projectile symbol the table is for ('p', 'd', 't', 'h' or 'a'). The ion Z and mass in the
        header must match ISOTOPIA's (constants.PARZ, constants.PARMASS) to the printed digits.

    Returns
    -------
    StoppingTable
    """
    path = Path(path)
    text = path.read_text(encoding="latin-1")
    k0 = PARSYM.index(particle)

    # Header: ion, density, phase, units, target element
    ion = _ION.search(text)
    if (
        ion is None
        or int(ion[1]) != PARZ[k0]
        or abs(float(ion[2]) - PARMASS[k0]) > 5e-7
    ):
        raise ValueError(
            f"{path.name}: ion in the header does not match ISOTOPIA's '{particle}' "
            f"(Z = {PARZ[k0]}, mass = {PARMASS[k0]:.6f} u)"
        )
    density = _DENSITY.search(text)
    units = _UNITS.search(text)
    if density is None or units is None:
        raise ValueError(
            f"{path.name}: not a SRIM stopping table (no density or units line)"
        )
    if units[1].replace(" ", "") != "eV/(1E15atoms/cm2)":
        raise ValueError(
            f"{path.name}: stopping units are '{units[1].strip()}'; "
            "tables must be in eV/(1E15 atoms/cm2)"
        )
    # Target composition block: the rows between "Target  Composition" and "Bragg Correction"
    block = text.partition("Target  Composition")[2].partition("Bragg Correction")[0]
    targets = _TARGET.findall(block)
    if len(targets) != 1:
        raise ValueError(
            f"{path.name}: target must be a single element (found {len(targets)})"
        )
    symbol, z = targets[0][0], int(targets[0][1])

    # Table rows: energy (converted to MeV), electronic and nuclear stopping per atom
    rows = _ROW.findall(text)
    if not rows:
        raise ValueError(f"{path.name}: no stopping rows found")
    energy = np.array([float(e) * _E_UNIT[u] for e, u, _, _ in rows])
    if np.any(np.diff(energy) <= 0):
        raise ValueError(f"{path.name}: energies are not increasing")
    return StoppingTable(
        particle=particle,
        Z=z,
        symbol=symbol,
        energy=energy,
        electronic=np.array([float(se) for _, _, se, _ in rows]),
        nuclear=np.array([float(sn) for _, _, _, sn in rows]),
        gas="Target is a GAS" in text,
        srim_density=float(density[1]),
        srim_atom_density=float(density[2]),
        path=path,
    )


def load_table(
    particle: str, Z: int, table_dir: str | Path | None = None
) -> StoppingTable:
    """Load the shipped SRIM table for an ISOTOPIA projectile in element Z.

    Parameters
    ----------
    particle : str
        ISOTOPIA projectile symbol: 'p', 'd', 't', 'h' (helium-3) or 'a'.
    Z : int
        Atomic number of the target element.
    table_dir : str or pathlib.Path, optional
        Folder holding ``<particle>/<particle>_in_<Symbol>.txt``; default: the shipped tables.

    Returns
    -------
    StoppingTable
    """
    if particle not in PARTICLES:
        raise ValueError(
            f"no SRIM tables for projectile '{particle}' (available: {', '.join(PARTICLES)})"
        )
    symbol = NUC[Z - 1].strip() if 1 <= Z <= len(NUC) else str(Z)
    path = Path(table_dir or TABLE_DIR) / particle / f"{particle}_in_{symbol}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"no SRIM table for {particle} in {symbol} (Z = {Z}): {path} is missing. "
            "Tables exist for every element from H to U (Z = 1-92); SRIM takes no heavier target."
        )
    table = read_table(path, particle)
    if table.Z != Z:
        raise ValueError(f"{path.name}: target Z = {table.Z}, expected {Z}")
    return table


def table_inputs(table_dir: str | Path | None = None) -> dict[int, dict[str, str]]:
    """Density and atomic mass each shipped table was made with (``index.csv``), by Z.

    Parameters
    ----------
    table_dir : str or pathlib.Path, optional
        Folder holding ``index.csv``; default: the shipped tables.

    Returns
    -------
    dict
        Z -> the index.csv row as strings: 'symbol', 'form', 'srim_phase', 'density_g_cm3'
        (g/cm3, 25 degC), 'target_mass_u' (u), 'density_flag', and the two file names.
    """
    with open(
        Path(table_dir or TABLE_DIR) / "index.csv", encoding="utf-8-sig", newline=""
    ) as fh:
        return {int(r["Z"]): r for r in csv.DictReader(fh)}
