"""Check SRIM stopping tables made by make_tables.py (all 92 elements, H to U).

Usage:  python scripts/srim/check_tables.py <particle> element_data.csv <output folder>

1. Header density equals the sheet density; the target mass implied by the header (N_A*rho/n)
   and by the footer factor for MeV/(mg/cm2) (0.602214/M) equals the sheet mass.
2. SRIM's gas flag is set exactly for srim_phase == gas.
3. Every file has the same energy grid, 1 keV to 250 MeV.
4. Electronic stopping vs ISOTOPIA's Bethe-Bloch (stoppingpower.f90 lines 42-58) at 10-250 MeV.
"""
import csv
import math
import re
import sys
from pathlib import Path

NA = 6.02214076e23
ION = {                                                          # constants.f90 lines 46, 48
    "p": (1, 1.00782495281),
    "d": (1, 2.01410195518),
    "t": (1, 3.01604916211),
    "h": (2, 3.01602879487),
    "a": (2, 4.00260293529),
}
UNIT = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}         # energy units -> MeV
NATURAL = list(range(1, 93))   # every element SRIM takes as a target, H to U

par, CSV, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
ion_z, ion_mass = ION[par]
sheet = {int(r["Z"]): r for r in csv.DictReader(open(CSV, encoding="utf-8-sig")) if r["default"] == "yes"}


def bethe(E, Z, A, rho):
    """stoppingpower.f90 (Leo 1994) in MeV/cm; A = sheet mass instead of the integer A."""
    pmass, emass = ion_mass * 931.49410242, 0.510998950
    beta = math.sqrt(E * (E + 2 * pmass)) / (E + pmass)
    Imean = Z * (9.76 + 58.8 * Z ** -1.19) * 1e-6
    eta = beta / math.sqrt(1 - beta ** 2)
    Wmax = 2 * emass * eta ** 2
    return 0.307075 * rho * Z / A * ion_z ** 2 / beta ** 2 * (0.5 * math.log(Wmax ** 2 / Imean ** 2) - beta ** 2)


def read(path):
    """Header density and atoms/cm3, footer MeV/(mg/cm2) factor, and (E [MeV], Se, Sn) rows."""
    t = path.read_text(encoding="latin-1")
    rho, n = map(float, re.search(r"Density =\s*(\S+) g/cm3 =\s*(\S+) atoms/cm3", t).groups())
    fac = float(re.search(r"^\s*(\S+)\s+MeV/\(mg/cm2\)", t, re.M)[1])
    rows = [(float(e) * UNIT[u], float(se), float(sn))
            for e, u, se, sn in re.findall(r"^\s*([\d.]+) (eV|keV|MeV|GeV)\s+(\S+)\s+(\S+)\s", t, re.M)]
    return t, rho, n, fac, rows


bad, grids, ratios = [], set(), {}
for z in NATURAL:
    r = sheet[z]
    sym = r["symbol"].strip()
    t, rho_h, n_h, fac, rows = read(OUT / par / f"{par}_in_{sym}.txt")
    rho, M = float(r["density_g_cm3"]), float(r["mass_u"])
    if abs(rho_h / rho - 1) > 5e-5: bad.append(f"{sym}: density {rho_h} vs {rho}")
    if abs(NA * rho_h / n_h / M - 1) > 1e-4: bad.append(f"{sym}: header mass {NA * rho_h / n_h:.4f} vs {M}")
    if abs(0.602214076 / fac / M - 1) > 1e-4: bad.append(f"{sym}: footer mass {0.602214076 / fac:.4f} vs {M}")
    if ("Target is a GAS" in t) != (r["srim_phase"] == "gas"): bad.append(f"{sym}: gas flag")
    if f"Mass = {ion_mass:.6f} amu" not in t: bad.append(f"{sym}: ion mass")
    grids.add(tuple(e for e, _, _ in rows))
    Se = {round(e, 6): s for e, s, _ in rows}
    ratios[(z, sym)] = [Se[E] * 1e-21 * n_h / bethe(E, z, M, rho_h) for E in (10.0, 50.0, 100.0, 250.0)]

g = next(iter(grids))
print(f"{par}: {len(ratios)} tables; problems: {bad or 'none'}")
print(f"{par}: {len(grids)} distinct energy grid(s); {len(g)} energies, {g[0] * 1e3:g} keV to {g[-1]:g} MeV")
for i, E in enumerate((10, 50, 100, 250)):
    hi = [(v[i], sym) for (z, sym), v in ratios.items() if z >= 13]
    print(f"{par}: SRIM/Bethe at {E:3d} MeV, Z >= 13: {min(hi)[0]:.3f} ({min(hi)[1]}) to {max(hi)[0]:.3f} ({max(hi)[1]})")
