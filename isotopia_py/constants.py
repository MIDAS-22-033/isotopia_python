"""Constants and basic particle/nuclide properties (constants.f90, reacinitial.f90)."""
from __future__ import annotations

from .fortran import S

NUMPAR = 6      # number of particles
NUMZ = 124      # number of elements
NUMISOM = 2     # number of isomers
NUMISO = 20     # maximum number of isotopes per element
NUMLINES = 100  # number of input lines
NUMA = 339      # number of masses
NUMEN = 1000000 # number of energies
NUMTIME = 100   # number of time points

# Particles: 0 gamma, 1 neutron, 2 proton, 3 deuteron, 4 triton, 5 helium-3, 6 alpha
PARNAME = ["gamma", "neutron", "proton", "deuteron", "triton", "helium-3", "alpha"]
PARSYM = ["g", "n", "p", "d", "t", "h", "a"]
PARZ = [0, 0, 1, 1, 1, 2, 2]
PARA = [0, 1, 1, 2, 3, 3, 4]
# parmass is real(dbl) in Fortran, but initialised from single-precision literals.
PARMASS = [float(S(m)) for m in
           (0.0, 1.00866491574, 1.00782495281, 2.01410195518, 3.01604916211,
            3.01602879487, 4.00260293529)]

# Element symbols, padded to two characters as in the Fortran (index Z-1).
NUC = [
    "H ", "He", "Li", "Be", "B ", "C ", "N ", "O ", "F ", "Ne",
    "Na", "Mg", "Al", "Si", "P ", "S ", "Cl", "Ar", "K ", "Ca", "Sc", "Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y ", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I ", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U ", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "B9", "C0", "C1", "C2", "C3", "C4",
]


def nuc(z: int) -> str:
    """Two-character element symbol for Z (Fortran nuc(Z))."""
    return NUC[z - 1]


# Isomer symbols, indexed by is = -1 (total), 0 (ground), 1, 2.
ISOCHAR = {-1: " ", 0: "g", 1: "m", 2: "n"}

# Fundamental constants
AMU = float(S(931.49410242))   # real(dbl) from a single-precision literal
EMASS = S(0.510998950)
AVOGADRO = S(6.02214076e23)
QELEM = S(1.602176634e-19)

# Default material densities in g/cm^3 (index Z-1)
RHOTABLE = [S(r) for r in (
    8.99e-5, 1.79e-4, 0.534, 1.85, 2.34, 2.267, 1.25e-3, 1.43e-3, 1.70e-3, 9.0e-3,
    0.97, 1.74, 2.7, 2.33, 1.82, 2.07, 3.21e-3, 1.78e-3, 0.86, 1.55, 2.99, 4.54, 6.11, 7.19, 7.43, 7.87, 8.9, 8.9, 8.96, 7.13,
    5.91, 5.32, 5.72, 4.79, 3.12, 3.75e-3, 1.53, 2.64, 4.47, 6.51,
    8.57, 10.22, 11.5, 12.37, 12.41, 12.02, 10.5, 8.65, 7.31, 7.31,
    6.68, 6.24, 4.93, 5.9e-3, 1.87, 3.59, 6.15, 6.77, 6.77, 7.01, 7.3, 7.52, 5.24, 7.9, 8.23, 8.55, 8.8, 9.07, 9.32, 6.9,
    9.84, 13.31, 16.65, 19.35, 21.04, 22.6, 22.4, 21.45, 19.32, 13.55,
    11.85, 11.35, 9.75, 9.3, 7.0, 9.73e-3, 1.87, 5.5, 10.07, 11.72,
    15.4, 18.95, 20.2, 19.84, 13.67, 13.5, 14.78, 15.1, 13.5, 13.5,
    13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5,
    13.5, 13.5, 13.5, 13.5,
)]

# Time units in seconds (single precision in the Fortran)
MINUTESEC = S(60.0)
HOURSEC = S(60.0) * MINUTESEC
DAYSEC = S(24.0) * HOURSEC
YEARSEC = S(365.25) * DAYSEC

VERSION_LINE = "    ISOTOPIA-2.23   (Version: July 9, 2026)"
