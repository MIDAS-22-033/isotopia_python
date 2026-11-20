"""Make SRIM stopping tables for ISOTOPIA projectiles in all 92 elements, H to U.

Usage:  python scripts/srim/make_tables.py <particle> element_data.csv "<SRIM folder>/SR Module" <output folder> [Z ...]
        Z ...    optional atomic numbers: make only these elements (default: all 92)
        particle = p (proton), d (deuteron), t (triton), h (helium-3) or a (alpha): ISOTOPIA's
                   symbols (constants.f90 line 45)

Writes <output folder>/<particle>/<particle>_in_<Symbol>.txt (e.g. p/p_in_Ni.txt) and the exact
SRIM input used, <particle>/inputs/<particle>_in_<Symbol>.SR.IN.

Elements: all 92 from H to U, the targets SRIM accepts. For the 84 with a CIAAW standard atomic
weight the target mass is that weight. For Tc, Pm, Po, At, Rn, Fr, Ra and Ac (no stable isotope)
it is the longest-lived isotope in the ISOTOPIA decay library. At and Fr densities are
estimates (Arblaster 2018).
Per element, from the default row of element_data.csv:
  target mass = mass_u (CIAAW standard atomic weight; abridged value where CIAAW gives an interval)
  density     = density_g_cm3 (25 degC; gases at 101.325 kPa)
  phase       = srim_phase (gas -> SRIM gas flag 1; Br and Hg liquids are run as solids)
Ion Z and mass: ISOTOPIA's parZ and parmass (constants.f90 lines 46 and 48), so the ion mass is the
atomic mass of H-1, H-2, H-3, He-3 or He-4 as ISOTOPIA writes it.
Units: eV/(1E15 atoms/cm2) (SR Module code 7). Energies: 1 keV to 250 MeV (ISOTOPIA's Ebeam
limit, checkvalue.f90 line 102) on SRIM's own grid.
"""
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---- ISOTOPIA projectiles (constants.f90 lines 45-48): symbol -> (Z, mass in u) -------------
ION = {
    "p": (1, "1.00782495281"),
    "d": (1, "2.01410195518"),
    "t": (1, "3.01604916211"),
    "h": (2, "3.01602879487"),     # helium-3
    "a": (2, "4.00260293529"),
}
EMIN_KEV, EMAX_KEV, UNITS = 1, 250000, 7
NATURAL = list(range(1, 93))   # every element SRIM takes as a target, H to U

par, CSV, SRDIR, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ONLY = [int(z) for z in sys.argv[5:]] or NATURAL
assert all(z in NATURAL for z in ONLY), "element not in the table set"
ion_z, ion_mass = ION[par]
(OUT / par / "inputs").mkdir(parents=True, exist_ok=True)

# ---- default row of each element from the data sheet ---------------------------------------
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8-sig")) if r["default"] == "yes"]
by_z = {int(r["Z"]): r for r in rows}
assert all(z in by_z for z in NATURAL), "data sheet is missing an element"

# ---- one SR Module run per element ----------------------------------------------------------
for z in ONLY:
    r = by_z[z]
    sym = r["symbol"].strip()
    name = r["crc_name"].split("(")[0].strip()          # SRIM label only, e.g. "Sulfur"
    gas = 1 if r["srim_phase"] == "gas" else 0
    fname = f"{par}_in_{sym}.txt"
    lines = [
        "---Stopping/Range Input Data (Number-format: Period = Decimal Point)",
        "---Output File Name", f'"{fname}"',
        "---Ion(Z), Ion Mass(u)", f"{ion_z}   {ion_mass}",
        "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.", f"{gas}    {r['density_g_cm3']}    1",
        "---Number of Target Elements", " 1 ",
        "---Target Elements: (Z), Target name, Stoich, Target Mass(u)", f'{z}   "{name}"   1   {r["mass_u"]}',
        "---Output Stopping Units (1-8)", f" {UNITS}",
        "---Ion Energy : E-Min(keV), E-Max(keV)", f" {EMIN_KEV}    {EMAX_KEV}",
    ]
    (SRDIR / "SR.IN").write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))   # SRIM wants CRLF
    (SRDIR / fname).unlink(missing_ok=True)
    cmd = [str(SRDIR / "SRModule.exe")] if os.name == "nt" else ["wine", "SRModule.exe"]  # Windows: native; Linux: Wine
    subprocess.run(cmd, cwd=SRDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    if not (SRDIR / fname).exists():
        sys.exit(f"SR Module produced no output for Z = {z} ({sym})")
    shutil.move(SRDIR / fname, OUT / par / fname)
    shutil.copy(SRDIR / "SR.IN", OUT / par / "inputs" / f"{par}_in_{sym}.SR.IN")

print(f"{len(ONLY)} {par} tables written to {OUT / par}")
