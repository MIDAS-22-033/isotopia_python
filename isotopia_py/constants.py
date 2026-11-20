"""Constants and basic particle/nuclide properties (constants.f90, reacinitial.f90)."""
from __future__ import annotations

from .fortran import S

NUMPAR = 6      # number of particles
NUMZ = 118      # number of elements (H to Og; heavier elements have not been made)
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
# parmass is real(dbl) in Fortran, but initialized from single-precision literals.
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
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def nuc(z: int) -> str:
    """Two-character element symbol for Z (Fortran nuc(Z))."""
    if z > NUMZ:
        raise ValueError(f"no element symbol for Z = {z}: ISOTOPIA covers Z = 1-{NUMZ}")
    return NUC[z - 1]


# Stopping-power models for the keyword 'stopping' (not in the Fortran): SRIM tables
# (isotopia_py/srim_tables) or the Fortran's Bethe-Bloch formula (stoppingpower.f90).
STOPPING_MODELS = ("srim", "bethe")

# Isomer symbols, indexed by is = -1 (total), 0 (ground), 1, 2.
ISOCHAR = {-1: " ", 0: "g", 1: "m", 2: "n"}

# Fundamental constants
AMU = float(S(931.49410242))   # real(dbl) from a single-precision literal
EMASS = S(0.510998950)
AVOGADRO = S(6.02214076e23)
QELEM = S(1.602176634e-19)

# Default material densities in g/cm^3 (index Z-1). The 'rho' keyword overrides them.
# None means there is no verified density for the element: the user must give
# 'rho' (checkvalue stops otherwise), so no unverified value is ever used.
#
# Values are from the group's element data table (element_data.csv), the row
# marked default for each element, at 25 degC (and 101.325 kPa for gases), except
# for these, whose values are NOT confirmed at 25 degC (the data table gives why):
#   - At: room temperature, an estimate (Arblaster 2018); no bulk astatine has been made;
#   - Fr: room temperature, an estimate (Arblaster 2018); no bulk francium has been made.
# Both Arblaster values are as given by the user from the book (ASM International, 2018): At on
# p. 604, Fr page not yet recorded. The data table gives the full citation.
# Solids, Hg and Rn: CRC Handbook of Chemistry and Physics, 107th ed. Br (liquid):
# Marcus, J. Chem. Thermodyn. 98, 317 (2016), eq. (3) evaluated at 298.15 K (the
# CRC value, 3.1028, has no temperature). Os, Ir: quadratic fits to Arblaster's
# density tables (Platinum Metals Rev. 57, 177 (2013); 54, 93 (2010)) over
# 200-400 K, evaluated at 298.15 K (the CRC values are at 20 degC). Derived values
# are given only to the precision of their source. Br and Hg are liquids at 25 degC.
# H, He, N, O, F, Ne, Cl, Ar, Kr and Xe are ideal-gas
# values at 25 degC, 101.325 kPa (rho = P*M/(R*T), diatomic M for H, N, O, F, Cl).
# Elements with several solid forms use the default form (see ALLOTROPES).
#
# Np, Pu, Cm, Cf (Z = 93, 94, 96, 98): CRC 107th ed. density table, 25 degC.
# Am, Bk (Z = 95, 97): X-ray densities, rho = n*M/(N_A*V_cell), for the
# longest-lived isotope (Am-243, Bk-247; the same isotope as the data table's
# mass column):
#   - Am: at 25 degC, from the lattice-parameter equations of McWhan et al.,
#     J. Inorg. Nucl. Chem. 24, 1025 (1962); used instead of the CRC density
#     table's 12, which is 13% lower and traces to an early sample whose crystal
#     cell could not be reproduced;
#   - Bk: from the lattice parameters at 298.15 K in Morss et al., The Chemistry of
#     the Actinide and Transactinide Elements, 3rd ed. (2006), Table 15.8 (the same
#     values as the CRC's); the CRC density table lists only beta-Bk (fcc, the phase above
#     977 degC); the default is alpha-Bk (dhcp, stable at 25 degC).
# Density scales with isotopic mass at fixed lattice.
#
# Es (Z = 99): X-ray density for Es-252 from the fcc lattice parameter at 298.15 K
# in Morss et al., The Chemistry of the Actinide and Transactinide Elements, 3rd
# ed. (2006), Table 15.8 (the CRC gives none); 3 significant figures, as a is.
#
# None (density must be given): Z = 100-118 (Fm to Og). No density has been
# measured for these, nor lattice parameters; the Fortran values were 13.5 placeholders.
RHOTABLE = [None if r is None else S(r) for r in (
    0.00008240, 0.0001636, 0.534, 1.85, 2.34, 2.2, 0.001145, 0.001308, 0.001553, 0.0008248,  # Z 1-10: H-Ne
    0.97, 1.74, 2.70, 2.3296, 2.16, 2.07, 0.002898, 0.001633, 0.89, 1.54,  # Z 11-20: Na-Ca
    2.99, 4.506, 6.0, 7.15, 7.3, 7.87, 8.86, 8.90, 8.96, 7.134,  # Z 21-30: Sc-Zn
    5.91, 5.3234, 5.75, 4.809, 3.0977, 0.003425, 1.53, 2.64, 4.47, 6.52,  # Z 31-40: Ga-Zr
    8.57, 10.2, 11, 12.1, 12.4, 12.0, 10.5, 8.69, 7.31, 7.287,  # Z 41-50: Nb-Sn
    6.68, 6.232, 4.933, 0.005366, 1.873, 3.62, 6.15, 6.770, 6.77, 7.01,  # Z 51-60: Sb-Nd
    7.26, 7.52, 5.24, 7.90, 8.23, 8.55, 8.80, 9.07, 9.32, 6.90,  # Z 61-70: Pm-Yb
    9.84, 13.3, 16.4, 19.3, 20.8, 22.587, 22.560, 21.5, 19.3, 13.53359,  # Z 71-80: Lu-Hg
    11.8, 11.3, 9.79, 9.20, 8.95, 0.009074, 2.458, 5, 10, 11.7,  # Z 81-90: Tl-Th
    15.4, 19.1, 20.2, 19.7, 13.79, 13.51, 14.67, 15.1, 8.81, None,  # Z 91-100: Pa-Fm
    None, None, None, None, None, None, None, None, None, None,  # Z 101-110: Md-Ds
    None, None, None, None, None, None, None, None,  # Z 111-118: Rg-Og
)]

# The Fortran rhotable (reacinitial.f90) for Z = 1-118, unchanged, including
# neon at 9.0e-3 and the 13.5 placeholders. Used only to check the port against
# the Fortran sample outputs (tests/test_samples.py); do not use it for new
# calculations.
RHOTABLE_FORTRAN = [S(r) for r in (
    8.99e-5, 1.79e-4, 0.534, 1.85, 2.34, 2.267, 1.25e-3, 1.43e-3, 1.70e-3, 9.0e-3,  # Z 1-10: H-Ne
    0.97, 1.74, 2.7, 2.33, 1.82, 2.07, 3.21e-3, 1.78e-3, 0.86, 1.55,  # Z 11-20: Na-Ca
    2.99, 4.54, 6.11, 7.19, 7.43, 7.87, 8.9, 8.9, 8.96, 7.13,  # Z 21-30: Sc-Zn
    5.91, 5.32, 5.72, 4.79, 3.12, 3.75e-3, 1.53, 2.64, 4.47, 6.51,  # Z 31-40: Ga-Zr
    8.57, 10.22, 11.5, 12.37, 12.41, 12.02, 10.5, 8.65, 7.31, 7.31,  # Z 41-50: Nb-Sn
    6.68, 6.24, 4.93, 5.9e-3, 1.87, 3.59, 6.15, 6.77, 6.77, 7.01,  # Z 51-60: Sb-Nd
    7.3, 7.52, 5.24, 7.9, 8.23, 8.55, 8.8, 9.07, 9.32, 6.9,  # Z 61-70: Pm-Yb
    9.84, 13.31, 16.65, 19.35, 21.04, 22.6, 22.4, 21.45, 19.32, 13.55,  # Z 71-80: Lu-Hg
    11.85, 11.35, 9.75, 9.3, 7.0, 9.73e-3, 1.87, 5.5, 10.07, 11.72,  # Z 81-90: Tl-Th
    15.4, 18.95, 20.2, 19.84, 13.67, 13.5, 14.78, 15.1, 13.5, 13.5,  # Z 91-100: Pa-Fm
    13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5,  # Z 101-110: Md-Ds
    13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5, 13.5,  # Z 111-118: Rg-Og
)]

# Elements with more than one solid form (allotrope) in the group's element data
# table (element_data.csv): CRC Handbook of Chemistry and Physics, 107th ed., at
# 25 degC; Bk: X-ray densities of both forms from the lattice parameters at
# 298.15 K, for Bk-247 (beta-Bk is metastable there). The first form listed
# is the default, whose density is
# the one in RHOTABLE; the others are reported so the user can pick one with the
# 'rho' keyword. Keyed Z: ((form, density), ...).
ALLOTROPES = {
    6: (("graphite", S(2.2)), ("diamond", S(3.513))),
    15: (("red", S(2.16)), ("white", S(1.823))),
    16: (("alpha, orthorhombic", S(2.07)), ("beta, monoclinic", S(2.00))),
    34: (("gray", S(4.809)), ("vitreous", S(4.28))),
    50: (("white", S(7.287)), ("gray", S(5.769))),
    97: (("alpha, dhcp", S(14.67)), ("beta, fcc (metastable)", S(13.15))),
}


def allotrope_warning(z: int, rho) -> list[str]:
    """Output lines telling the user that element Z has allotropes with other
    densities, and which density ``rho`` [g/cm^3] is being used. Empty when Z
    has only one tabulated form.
    """
    forms = ALLOTROPES.get(z)
    if not forms:
        return []
    lines = [f" ISOTOPIA-warning: other densities were found for {nuc(z).strip()} (allotropes):"]
    for form, r in forms:
        tag = "  (default)" if r == rho else ""
        lines.append(f"     {form:<22s}{float(r):8.3f} g/cm^3{tag}")
    lines.append(f"   Using {float(rho):.3f} g/cm^3 from the density table;"
                 " set the keyword 'rho' to use another value.")
    return lines

# Time units in seconds (single precision in the Fortran)
MINUTESEC = S(60.0)
HOURSEC = S(60.0) * MINUTESEC
DAYSEC = S(24.0) * HOURSEC
YEARSEC = S(365.25) * DAYSEC

VERSION_LINE = "    ISOTOPIA-2.23   (Version: July 9, 2026)"
