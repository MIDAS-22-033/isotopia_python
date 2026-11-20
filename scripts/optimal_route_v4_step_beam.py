"""Search for the optimal production route of a desired isotope (DI).

For a DI with mass number A, the script runs isotopia_py for every candidate
target with mass number A-5..A+5 and every beam energy on a linear grid. Each
route reports the mass of the DI produced (g) at end of irradiation and at end
of cooling, and the initial mass production rate; routes are ranked by the
--metric chosen (default: mass at end of cooling). Stable and radioactive DIs
are handled the same way, because the masses come from ISOTOPIA's atom counts.

    python scripts/optimal_route.py Tc99_m1 --projectile p --de 1 --thickness 0.05
    python scripts/optimal_route.py Cu64 -p p d --de 0.5 --exclude blacklist.txt --jobs 4

Outputs (in --out, default runs/optimal_route/<DI>):
    scan.csv     one row per run (route, Ebeam, Eback, masses, status)
    summary.txt  assumptions, best route per target, overall best, warnings

How each projectile is treated
    p, d, t, h, a  Ebeam scanned on the grid; the target thickness is passed to
                   ISOTOPIA (keyword 'thickness'), which solves Eback with its own
                   stopping-power model (SRIM by default, see isotopia_py.rates).
    g              Electron energy scanned, but ISOTOPIA only has bremsstrahlung
                   spectra for integer 1-200 MeV, so the grid is reduced to
                   distinct integer energies >= 1 MeV.
    n              Reactor spectrum (files/neutron_spectra/n-hfr.dat) does not
                   depend on Ebeam: one run per target, no energy scan.

Exclusion list (--exclude FILE): one nuclide per line, '#' starts a comment.
A route is excluded if its target, or a parent that feeds the DI by decay in
the ISOTOPIA model (same A, Z-1 beta- or Z+1 beta+), is on the list.
Names: Tc99 (any state), Tc99_g (ground), Tc99_m1 / Tc99_m2 (isomers).

Rarity weighting (--rarity WORKBOOK): each route's score is its mass metric times
(1 + Delta)/2, with Delta = rarity(DI) - rarity(target) from the team's price /
obtainability workbook (see `RarityIndex`). Routes whose target is rarer than the
DI are kept but scored lower and flagged. --exclude is applied independently.

Needs ISOTOPIA_DIR (and ISOTOPIA_LIBS unless isotopia.libs sits next to it),
exactly as for `isotopia_py.run`.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isotopia_py.constants import AVOGADRO, NUC, NUMTIME, PARA, PARMASS, PARSYM, PARZ, nuc  # noqa: E402
from isotopia_py.fortran import endf_float  # noqa: E402
from isotopia_py.inputdeck import machine  # noqa: E402
from isotopia_py.run import run  # noqa: E402
from isotopia_py.state import IsotopiaError, State  # noqa: E402

MASS_WINDOW = 5          # targets have mass number A_DI - MASS_WINDOW .. A_DI + MASS_WINDOW
ZDEPTH_DEFAULT = 10      # ISOTOPIA default Zdepth (input.f90); limits which Z it computes
ADEPTH_DEFAULT = 20      # ISOTOPIA default Adepth (input.f90); limits which A it computes
NEUTRON_MASS_U = PARMASS[1]   # u; neutron mass from isotopia_py.constants (constants.f90)

Nuclide = tuple[int, int, int | None]   # (Z, A, I); I = None means "any state"


# ----------------------------------------------------------------------------
# Nuclide names and lists
# ----------------------------------------------------------------------------

_NAME = re.compile(r"^([A-Z][a-z]?)(\d{1,3})(?:_(g|m1|m2))?$")
_STATE = {None: None, "g": 0, "m1": 1, "m2": 2}


def parse_nuclide(name: str) -> Nuclide:
    """Parse 'Tc99', 'Tc99_g', 'Tc99_m1' into (Z, A, I); bare names give I = None."""
    m = _NAME.match(name.strip())
    if not m:
        raise ValueError(f"cannot parse nuclide name '{name}' (expected e.g. Tc99, Tc99_g, Tc99_m1)")
    symbol, mass, state = m.groups()
    symbols = [s.strip() for s in NUC]
    if symbol not in symbols:
        raise ValueError(f"unknown element symbol '{symbol}' in '{name}'")
    return symbols.index(symbol) + 1, int(mass), _STATE[state]


def nuclide_name(Z: int, A: int, I: int | None) -> str:
    """Display name in the project convention: Tc99, Tc99_g, Tc99_m1."""
    suffix = {None: "", -1: "", 0: "_g", 1: "_m1", 2: "_m2"}[I]
    return f"{nuc(Z).strip()}{A}{suffix}"


def read_nuclide_list(path: str | os.PathLike) -> set[Nuclide]:
    """Read a nuclide list file (exclusion now; the same format serves a warning list later).

    One or more names per line separated by spaces or commas; '#' starts a comment.
    """
    entries: set[Nuclide] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            for token in raw.split("#", 1)[0].replace(",", " ").split():
                try:
                    entries.add(parse_nuclide(token))
                except ValueError as exc:
                    raise ValueError(f"{path}, line {lineno}: {exc}") from None
    return entries


def listed(Z: int, A: int, is_: int, entries: set[Nuclide]) -> Nuclide | None:
    """Return the list entry matching ISOTOPIA state (Z, A, is_), or None.

    is_ = -1 is ISOTOPIA's 'total' (all states), so it matches any entry for Z, A.
    A bare entry (I = None) matches every state.
    """
    for eZ, eA, eI in entries:
        if eZ == Z and eA == A and (eI is None or is_ == -1 or eI == is_):
            return (eZ, eA, eI)
    return None


# ----------------------------------------------------------------------------
# Candidate targets
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    Z: int
    A: int
    stable: bool

    @property
    def name(self) -> str:
        return nuclide_name(self.Z, self.A, None)


def natural_masses(path: str, Z: int) -> set[int]:
    """Mass numbers with nonzero abundance in files/abundance/zZZZ (abundance.f90 format)."""
    abfile = Path(path) / "files" / "abundance" / f"z{Z:03d}"
    masses: set[int] = set()
    if abfile.exists():
        for line in abfile.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.ljust(19)
            if line[4:8].strip() and line[8:19].strip() and float(line[8:19]) > 0:
                masses.add(int(line[4:8]))
    return masses


def candidate_targets(di: Nuclide, k0: int, path: str, crosspath: str,
                      include_radioactive: bool, warnings: list[str]) -> list[Target]:
    """Library targets with A_DI - 5 <= A <= A_DI + 5 whose ISOTOPIA window contains the DI.

    ISOTOPIA computes residuals for Z from Zcomp + 1 down to Zcomp - Zdepth and
    for A from Acomp down to Acomp - Adepth, with Zcomp = Ztarget + Zprojectile,
    Acomp = Atarget + Aprojectile, Zdepth = min(10, Ztarget - 1) and
    Adepth = min(20, Atarget - 1) (reacinitial.f90, checkvalue.f90).

    A reaction can raise the mass number by at most Aprojectile (capture of
    the whole projectile), so a lighter target can only reach the DI if
    Atarget + Aprojectile >= A_DI. Targets that fail either window cannot
    produce the DI in the code; they are not run, and lighter targets
    dropped this way are reported in `warnings`.
    """
    Zdi, Adi, _ = di
    libdir = Path(crosspath) / PARSYM[k0]
    if not libdir.is_dir():
        raise SystemExit(f"cross-section library not found: {libdir}")
    pattern = re.compile(r"^([A-Z][a-z]?)(\d{3})(m|n)?$")
    symbols = [s.strip() for s in NUC]
    targets = []
    too_light: list[str] = []
    for entry in sorted(os.listdir(libdir)):
        m = pattern.match(entry)
        if not m or m.group(1) not in symbols:
            continue
        Z, A = symbols.index(m.group(1)) + 1, int(m.group(2))
        if abs(A - Adi) > MASS_WINDOW:
            continue
        if m.group(3):
            warnings.append(f"{PARSYM[k0]}: isomeric target {entry} skipped "
                            "(isotopia_py input deck cannot take isomeric targets)")
            continue
        if (Z, A) == (Zdi, Adi) and di[2] in (0, None):
            # The DI's own ground state as target: ISOTOPIA's inventory for it is the
            # unreacted target (it only depletes), not production. An isomeric DI from
            # its own ground state (e.g. Tc-99 -> Tc-99m by inelastic scattering) is kept.
            warnings.append(f"{PARSYM[k0]}: target {entry} skipped (it is the DI itself)")
            continue
        Zcomp, Acomp = Z + PARZ[k0], A + PARA[k0]
        if not Zcomp - min(ZDEPTH_DEFAULT, Z - 1) <= Zdi <= Zcomp + 1:
            continue
        if not Acomp - min(ADEPTH_DEFAULT, A - 1) <= Adi <= Acomp:
            if Adi > Acomp:
                too_light.append(entry)
            continue
        stable = A in natural_masses(path, Z)
        if not stable and not include_radioactive:
            continue
        targets.append(Target(Z, A, stable))
    if too_light:
        warnings.append(f"{PARSYM[k0]}: {len(too_light)} lighter targets skipped, A + {PARA[k0]} < "
                        f"{Adi} so they cannot reach the DI mass: {', '.join(too_light)}")
    return targets


# ----------------------------------------------------------------------------
# Atomic mass of the DI (for atoms -> grams)
# ----------------------------------------------------------------------------

def molar_mass(path: str, Z: int, A: int) -> float:
    """Molar mass of nuclide (Z, A) in g/mol, from ISOTOPIA's decay library.

    Every nuclide in files/decay/zZZZ (JEFF-3.1.1, ENDF-6 format), stable ones
    included, starts with an MF1/MT451 record whose first line holds ZA and
    AWR, the atomic mass in units of the neutron mass (ENDF-6 manual, Sec. 1.1).
    The atomic mass is AWR * m_n, and in g/mol it equals the mass in u.

    Raises
    ------
    LookupError
        If the file or the nuclide's record is missing (data must come from
        the library, so the search stops and reports it).
    """
    decayfile = Path(path) / "files" / "decay" / f"z{Z:03d}"
    if not decayfile.exists():
        raise LookupError(f"decay file {decayfile} not found")
    za = 1000 * Z + A
    with open(decayfile, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # First line of an MF1/MT451 section: ZA in columns 1-11, AWR in 12-22
            if line[71:80] == "1451    1" and round(endf_float(line[0:11])) == za:
                return endf_float(line[11:22]) * NEUTRON_MASS_U
    raise LookupError(f"no record (AWR) for {nuclide_name(Z, A, None)} in {decayfile}")


# ----------------------------------------------------------------------------
# Rarity index (team workbook Element_Prices_per_Gram_USD.xlsx)
# ----------------------------------------------------------------------------

RARITY_BASES = ("price", "obtainability")
_ISOTOPE_IN_TEXT = re.compile(r"-(\d{1,3})(m)?(?![\d%])")   # "Technetium-99m", "93% U-235"


def _num(v) -> float:
    """Excel N(): numbers pass through, anything else (blank, text) is 0."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _isotope_from_text(text: str) -> tuple[int, int] | None:
    """(A, I) named in a workbook label, e.g. 'Technetium-99m' -> (99, 1); None if absent."""
    if "deuterium" in text.lower():
        return 2, 0
    m = _ISOTOPE_IN_TEXT.search(text)
    return (int(m.group(1)), 1 if m.group(2) else 0) if m else None


class RarityIndex:
    """Rarity index in [0, 1] (0 = least rare) per nuclide, read from the team workbook.

    The workbook stores formulas without cached values, so they are evaluated here
    from the raw columns, exactly as written in the sheet:

    price          'Price per gram' column I: (ln F - ln F_min) / (ln F_max - ln F_min),
                   F = average market price (USD/g); min and max over every row with
                   F > 0, isotope rows included.
    obtainability  'US supply ranking' column E: H/100 * (W + (1 - W) * K/100), with
                   H = import reliance used (%), K = China + Russia share (%), taken as
                   min(100, China + Russia), or the sheet's assumed share when both are
                   blank; W and the assumed share come from cells E3 and E4. Only rows
                   7-85 are ranked (the sheet's own lookup range).

    Lookup order, following the team's 'Rarity Index Structure' diagram, with every
    fallback recorded as a flag: isotope-specific price row if one exists, else the
    element row; if the chosen basis has no value, the other basis; if neither, 0.
    Obtainability is defined per element in the workbook, so it never has an
    isotope-level value and that is not flagged.
    """

    def __init__(self, path: str | os.PathLike):
        try:
            import openpyxl  # optional: only needed when --rarity is used
        except ImportError:
            raise SystemExit("--rarity needs openpyxl (pip install openpyxl)") from None
        wb = openpyxl.load_workbook(path, data_only=False, read_only=True)

        # Price sheet: columns located by header text, not position
        rows = list(wb["Price per gram"].iter_rows(values_only=True))
        h = {str(name).strip(): k for k, name in enumerate(rows[0]) if name}
        cz, crow, cform = h["Z"], h["Row type"], h["Form / isotope priced"]
        cprice, cregion = h["Average market price (USD per gram)"], h["Market / region of price"]
        self.element_price: dict[int, tuple[float, str]] = {}
        self.isotope_price: dict[tuple[int, int, int], tuple[float, str]] = {}
        positive = []
        for r in rows[1:]:
            if not isinstance(r[cz], int) or _num(r[cprice]) <= 0:
                continue
            Z, price, region = r[cz], _num(r[cprice]), str(r[cregion])
            positive.append(price)
            kind = str(r[crow])
            if kind.startswith("Element"):
                self.element_price.setdefault(Z, (price, region))
                if "main isotope" in kind:       # e.g. 'Element — main isotope (Pu-239)'
                    iso = _isotope_from_text(kind)
                    if iso:
                        self.isotope_price[(Z, *iso)] = (price, region)
            elif kind.startswith("Isotope variant"):
                iso = _isotope_from_text(str(r[cform])) or _isotope_from_text(kind)
                if iso:
                    self.isotope_price[(Z, *iso)] = (price, region)
        self._ln_min, self._ln_max = np.log(min(positive)), np.log(max(positive))

        # Obtainability sheet: settings in E3/E4, header on row 6, ranked rows 7-85
        ws = wb["US supply ranking"]
        cells = list(ws.iter_rows(min_row=1, max_row=85, values_only=True))
        W, assumed = _num(cells[2][4]), _num(cells[3][4])
        hdr = {str(name).strip(): k for k, name in enumerate(cells[5]) if name}
        ch = hdr["Import reliance used (%)"]
        cchina, crussia = hdr["China share of imports (%)"], hdr["Russia share of imports (%)"]
        self.obtainability: dict[int, float] = {}
        for r in cells[6:85]:
            if not isinstance(r[1], int) or not isinstance(r[ch], (int, float)):
                continue
            china, russia = r[cchina], r[crussia]
            K = assumed * 100 if china is None and russia is None else min(100.0, _num(china) + _num(russia))
            self.obtainability[r[1]] = r[ch] / 100 * (W + (1 - W) * K / 100)

    def _price(self, Z: int, A: int, I: int, flags: list[str]) -> float | None:
        name = nuclide_name(Z, A, I)
        hit = self.isotope_price.get((Z, A, I))
        if hit is None and Z in self.element_price:
            hit = self.element_price[Z]
            flags.append(f"{name}: element-level price used (no isotope row)")
        if hit is None:
            return None
        price, region = hit
        if not region.startswith("United States"):
            flags.append(f"{name}: price is from '{region}', not the U.S. market")
        return self.log_normalised(price)

    def log_normalised(self, price: float) -> float:
        """Sheet column I: price (USD/g) on a log scale between the sheet's min and max -> [0, 1]."""
        return float((np.log(price) - self._ln_min) / (self._ln_max - self._ln_min))

    def lookup(self, Z: int, A: int, I: int, basis: str) -> tuple[float, list[str]]:
        """Rarity index (0-1) of nuclide (Z, A, I) on `basis`, plus fallback flags."""
        flags: list[str] = []
        name = nuclide_name(Z, A, I)
        get = {"price": lambda: self._price(Z, A, I, flags),
               "obtainability": lambda: self.obtainability.get(Z)}
        other = RARITY_BASES[1 - RARITY_BASES.index(basis)]
        value = get[basis]()
        if value is None:
            value = get[other]()
            if value is not None:
                flags.append(f"{name}: no {basis} value, used {other}")
        if value is None:
            flags.append(f"{name}: no price or obtainability value, assumed 0")
            value = 0.0
        return value, flags


def apply_rarity(rows: list[dict], di: Nuclide, index: RarityIndex, basis: str, col: str,
                 warnings: list[str]) -> None:
    """Add rarity columns and score = row[col] * weight to every successful row (in place).

    Rarity depends only on the target (P1, ground state) and the DI, so it is looked
    up once per target; each fallback flag and each target rarer than the DI is
    reported once in `warnings`.
    """
    r_di, di_flags = index.lookup(*di, basis)
    warnings.extend(f"rarity: {f}" for f in di_flags)
    per_target: dict[str, tuple[float, float, float, list[str]]] = {}
    for row in rows:
        if row["status"] != "ok" or row.get(col) is None:
            continue
        name = row["target"]
        if name not in per_target:
            Z, A, _ = parse_nuclide(name)
            r_t, flags = index.lookup(Z, A, 0, basis)
            delta, weight = rarity_weight(r_t, r_di)
            per_target[name] = (r_t, delta, weight, flags)
            warnings.extend(f"rarity: {f}" for f in flags)
            if delta < 0:
                warnings.append(f"rarity: target {name} is rarer than the DI "
                                f"(R = {r_t:.3f} vs {r_di:.3f}); kept with weight {weight:.3f}")
        r_t, delta, weight, flags = per_target[name]
        row.update(R_target=r_t, R_DI=r_di, delta_R=delta, rarity_weight=weight,
                   score=row[col] * weight, rarity_flags="; ".join(di_flags + flags))


def rarity_weight(r_target: float, r_di: float) -> tuple[float, float]:
    """Delta = R_DI - R_target in [-1, 1] and weight (1 + Delta) / 2 in [0, 1].

    Delta < 0 means the starting material is rarer than the product; the route is
    kept but its score is scaled down (weight 0 when the target is maximally rare
    and the product is not rare at all).
    """
    delta = r_di - r_target
    return delta, (1.0 + delta) / 2.0


# ----------------------------------------------------------------------------
# One ISOTOPIA run
# ----------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Everything a worker needs for one run (picklable for multiprocessing)."""
    di: Nuclide
    k0: int
    target: Target
    ebeam: float | None          # MeV; None for neutrons
    eback: float | None          # MeV; filled from the run (charged particles), None before
    lines: list[str]
    exclude: set[Nuclide] = field(default_factory=set)
    molar_mass: float = 0.0      # g/mol of the DI, from molar_mass()


def di_key(st: State, di: Nuclide) -> tuple[int, int, int] | None:
    """ISOTOPIA key for the DI. Ground state is stored as 'total' (-1) when
    the library has no isomer for that Z, A. Returns None if the state is absent."""
    Z, A, I = di
    n_isomer = st.Nisomer[(Z, A)]
    if I in (0, None) and n_isomer == -1:
        return (Z, A, -1)
    if I is not None and I > n_isomer:
        return None
    return (Z, A, -1 if I is None else I)


def precursors(st: State, di: Nuclide) -> list[tuple[int, int, int]]:
    """Parents that feed the DI by decay in the model (kinetics.equations, step 2):
    same A, Z-1 with rtyp 1 (beta-) or Z+1 with rtyp 2 (beta+), produced in this run."""
    Z, A, _ = di
    found = []
    for isob, rt in ((-1, 1), (1, 2)):
        Zp = Z + isob
        for is_p in range(-1, st.Nisomer[(Zp, A)] + 1):
            pkey = (Zp, A, is_p)
            if st.rtyp[pkey] == rt and float(st.reaction_rate[pkey]) > 0:
                found.append(pkey)
    return found


BEAM_STOPS = "beam stops in the target"


def stop_note(st: State) -> str:
    """Note for a run whose beam stops inside the target, else "".

    With the 'thickness' input ISOTOPIA (rates._setup_target) then sets Eback to the
    lowest stopping-table energy and counts only the depth the beam reaches (Leff [cm])
    as the irradiated target.
    """
    if st.k0 > 1 and getattr(st, "beam_stops", False):
        return f"{BEAM_STOPS} after {float(st.Leff):.4g} cm; only that depth is irradiated"
    return ""


def stopping_warnings(rows: list[dict]) -> list[str]:
    """One warning per projectile + target whose runs used the Bethe-Bloch fallback.

    ISOTOPIA (rates._setup_target) uses SRIM tables where they exist (p, d, t, h, a in
    Z = 1-92) and otherwise Bethe-Bloch, writing a warning to its own output, which the
    optimizer does not keep. This brings that warning into summary.txt.

    Parameters
    ----------
    rows : list of dict
        Rows from run_one or run_rates; each carries "stopping" ("srim", "bethe" or None),
        "projectile" and "target".

    Returns
    -------
    list of str
        Warnings, in the order the targets first appear.
    """
    bethe = dict.fromkeys((r["projectile"], r["target"]) for r in rows if r.get("stopping") == "bethe")
    return [f"{p}+{t}: no SRIM stopping table (SRIM covers H to U, Z = 1-92); Eback and Leff "
            "come from the Bethe-Bloch formula, which is not valid below the energy of "
            "its maximum (it depends on projectile and target), so runs where the beam would stop in the target fail"
            for p, t in bethe]


def run_one(spec: RunSpec) -> dict:
    """Run isotopia_py once and extract the DI masses; never raises."""
    t = spec.target
    row = {"projectile": PARSYM[spec.k0], "target": t.name, "stable": t.stable,
           "Ebeam_MeV": spec.ebeam, "Eback_MeV": spec.eback, "Leff_cm": None, "stopping": None,
           "m_rate": None, "m_rate_unit": rate_unit(spec.k0),
           "m_EOI_g": None, "m_EOC_g": None, "status": "ok", "note": ""}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            st = run(spec.lines, out=io.StringIO(), outdir=tmp)
    except (IsotopiaError, OSError, ValueError) as exc:
        row.update(status="error", note=" ".join(str(exc).split()))
        return row
    if spec.k0 > 1:                                   # Eback solved by ISOTOPIA from the thickness
        row.update(Eback_MeV=float(st.Eback), Leff_cm=float(st.Leff), stopping=st.stopping_model,
                   note=stop_note(st))

    # Exclusion: precursors that feed the DI in this run
    for pkey in precursors(st, spec.di):
        hit = listed(*pkey, spec.exclude)
        if hit:
            row.update(status="excluded", note=f"precursor {nuclide_name(*hit)} on exclusion list")
            return row

    key = di_key(st, spec.di)
    if key is None:
        row.update(status="not_produced",
                   note="no positive DI cross section between Eback and Ebeam for this target")
        return row

    # Masses from ISOTOPIA's atom counts N_DI(t) (st.Niso, kinetics.equations):
    # m = N * M / N_A. Atom counts exist for stable and radioactive nuclides alike.
    N = st.Niso[key]
    g_per_atom = spec.molar_mass / float(AVOGADRO)
    i_eoc = NUMTIME if st.Tco > 0 else st.Ntime   # with Tcool = 0 there is no cooling grid
    row["m_EOI_g"] = float(N[st.Ntime]) * g_per_atom
    row["m_EOC_g"] = float(N[i_eoc]) * g_per_atom

    # Initial mass production rate: slope of N over the first time step, the
    # same finite difference ISOTOPIA uses for its activity yield (kinetics.activities),
    # normalised the same way: per mA of beam (charged) or per g of target (n, g).
    dt_h = float(st.Tgrid[1] - st.Tgrid[0]) / 3600.0
    per = float(st.Ibeam) * 1e3 if spec.k0 > 1 else float(st.M_target)   # mA or g
    row["m_rate"] = float(N[1] - N[0]) * g_per_atom / (per * dt_h)
    return row


def rate_unit(k0: int) -> str:
    """Unit of m_rate for projectile index k0 (matches ISOTOPIA's yield normalisation)."""
    return "g/(mA h)" if k0 > 1 else "g/(g h)"


# ----------------------------------------------------------------------------
# Building the scan
# ----------------------------------------------------------------------------

def energy_grid(emin: float, emax: float, de: float) -> np.ndarray:
    """Linear grid emin, emin + de, ... <= emax (MeV)."""
    n = int(np.floor((emax - emin) / de + 1e-9))
    return emin + de * np.arange(n + 1)


def base_lines(t: Target, k0: int, args) -> list[str]:
    """ISOTOPIA input keywords shared by every run on this target (manual Sec. 6)."""
    lines = [f"projectile {PARSYM[k0]}", f"element {nuc(t.Z).strip()}", f"mass {t.A}",
             f"Ibeam {args.ibeam}", f"Tirrad {args.tirrad}", f"Tcool {args.tcool}",
             f"thickness {args.thickness}"]     # charged particles: ISOTOPIA solves Eback from it
    if k0 == 1 and args.flux is not None:
        lines.append(f"fluxtotal {args.flux}")
    return lines


def build_specs(di: Nuclide, k0: int, targets: list[Target], grid: np.ndarray,
                args, exclude: set[Nuclide], warnings: list[str]) -> tuple[list[RunSpec], list[dict]]:
    """All runs for one projectile, plus pre-filled rows for routes skipped before running."""
    specs, skipped = [], []
    p = PARSYM[k0]
    for t in targets:
        hit = listed(t.Z, t.A, 0, exclude)
        if hit:
            skipped.append({"projectile": p, "target": t.name, "stable": t.stable,
                            "status": "excluded", "note": f"target {nuclide_name(*hit)} on exclusion list"})
            continue
        lines = base_lines(t, k0, args)
        if k0 == 1:                                   # neutrons: spectrum independent of Ebeam
            specs.append(RunSpec(di, k0, t, None, None, lines, exclude))
        elif k0 == 0:                                 # photons: integer spectra 1..200 MeV
            for e in sorted({int(e) for e in grid if e >= 1.0}):
                specs.append(RunSpec(di, k0, t, float(e), None, lines + [f"Ebeam {e}."], exclude))
        else:                                         # charged particles: Ebeam only; ISOTOPIA
            for e in grid:                            # solves Eback from the thickness
                specs.append(RunSpec(di, k0, t, float(e), None,
                                     lines + [f"Ebeam {e:.6f}"], exclude))
    if k0 == 1:
        warnings.append("n: reactor spectrum does not depend on Ebeam; one run per target, no energy scan")
    if k0 == 0:
        warnings.append("g: bremsstrahlung spectra exist only for integer 1-200 MeV; "
                        "grid reduced to distinct integer energies >= 1 MeV")
    return specs, skipped


# ----------------------------------------------------------------------------
# Step beam: energy that changes in steps during irradiation (--steps N)
#
# NEW PHYSICS relative to ISOTOPIA. ISOTOPIA's production equations (manual
# Sec. 4, equations.f90) assume one constant beam energy. Here the beam is a step
# function: N steps, each at its own energy, whose switch times and energies are
# optimised. Within a step every rate is constant, so the inventory obeys the
# linear system dN/dt = M N, whose exact solution over a step of length dt is
# N(t + dt) = expm(M dt) N(t) (Moler & Van Loan, SIAM Review 45 (2003) 3;
# scipy's expm: Al-Mohy & Higham, SIAM J. Matrix Anal. Appl. 31 (2009) 970).
# This is the matrix-exponential depletion method of ORIGEN (Croff, ORNL-5621,
# 1983) and reduces to the Bateman solution (Proc. Camb. Phil. Soc. 15 (1910)
# 423) for a linear chain.
#
# Decay uses FULL chains from the decay library: every decay mode, branching
# ratio and daughter isomeric state in files/decay/zZZZ (ENDF-6 MF8/MT457, ENDF-6
# Formats Manual BNL-90365-2009 Sec. 8.4), not ISOTOPIA's one-step beta feeding.
# Reaction rates still come from ISOTOPIA (one run per grid energy), linearly
# interpolated between grid energies. Products are not hit again by the beam.
# ----------------------------------------------------------------------------

StateKey = tuple[int, int, int]   # (Z, A, I), I = 0 ground, 1, 2 isomers

# Change in (Z, A) for each ENDF decay-mode digit (RTYP); 6 (fission) has no tracked daughter
_DECAY_DIGIT = {"1": (1, 0), "2": (-1, 0), "3": (0, 0), "4": (-2, -4), "5": (0, -1), "7": (-1, -1)}


class DecayLibrary:
    """Decay constants and branches for every state in files/decay/zZZZ, read on demand.

    get(state) -> (lambda [s^-1], [(daughter state or None, branching ratio)]), or
    None if the state has no record. A daughter of None is a loss (fission, or a
    decay-mode code this reader does not map), noted in `self.notes`.
    """

    def __init__(self, path: str):
        self.dir = Path(path) / "files" / "decay"
        self._z: dict[int, dict[tuple[int, int], tuple[float, list]]] = {}
        self.notes: list[str] = []

    @staticmethod
    def _daughter(Z: int, A: int, rtyp: float, rfs: int) -> StateKey | None:
        """Daughter of a decay mode: RTYP digits are successive emissions (1.5 = beta- then n)."""
        digits = f"{rtyp:.4f}".replace(".", "").rstrip("0")
        for d in digits:
            if d not in _DECAY_DIGIT:
                return None
            dz, da = _DECAY_DIGIT[d]
            Z, A = Z + dz, A + da
        return (Z, A, rfs)

    def _load(self, Z: int) -> dict:
        """Parse every MF8/MT457 section of zZZZ: HEAD (ZA, LISO), half-life and NC,
        the spin/parity record with NDK, then NDK lines of (RTYP, RFS, Q, dQ, BR, dBR)."""
        table: dict[tuple[int, int], tuple[float, list]] = {}
        f = self.dir / f"z{Z:03d}"
        if not f.exists():
            return table
        sections, cur = [], None
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if line[70:72].strip() != "8" or line[72:75] != "457":
                continue
            if line[75:80].strip() == "1":
                cur = []
                sections.append(cur)
            if cur is not None:
                cur.append([endf_float(line[11 * k:11 * k + 11]) for k in range(6)])
        for sec in sections:
            A = round(sec[0][0]) % 1000
            liso = round(sec[0][3])
            thalf, nc = sec[1][0], round(sec[1][4])
            k = 2 + -(-nc // 6)                      # skip the NC average-energy values
            ndk = round(sec[k][5])
            lam = 0.0 if thalf <= 0 else float(np.log(2.0) / thalf)
            branches = []
            if lam > 0:
                for rtyp, rfs, _, _, br, _ in sec[k + 1:k + 1 + ndk]:
                    branches.append((self._daughter(Z, A, rtyp, round(rfs)), br))
                    if branches[-1][0] is None:
                        self.notes.append(f"{nuclide_name(Z, A, liso)}: decay mode RTYP={rtyp:g} "
                                          f"(BR {br:.3g}) has no tracked daughter; treated as a loss")
            table[(A, liso)] = (lam, branches)
        return table

    def get(self, state: StateKey) -> tuple[float, list] | None:
        Z, A, I = state
        if Z not in self._z:
            self._z[Z] = self._load(Z)
        return self._z[Z].get((A, I))


def build_network(target: StateKey, di: StateKey, produced: set[StateKey], decay: DecayLibrary,
                  warnings: list[str], max_states: int = 5000) -> list[StateKey] | None:
    """States that can affect the DI: the target (index 0), every produced state from
    which the DI is reachable by decay, the intermediate states on those paths, and
    the DI. Returns None if the DI cannot be reached at all.

    States without decay data are treated as stable (chain truncated, with a warning);
    a missing isomer record of a daughter falls back to that nuclide's ground state.
    """
    children: dict[StateKey, list[StateKey]] = {}
    seen, todo = set(), list(produced | {target})
    while todo:                                     # forward closure along decay branches
        s = todo.pop()
        if s in seen:
            continue
        seen.add(s)
        if len(seen) > max_states:
            warnings.append(f"step beam: decay network capped at {max_states} states (chains truncated)")
            break
        rec = decay.get(s)
        if rec is None:
            warnings.append(f"step beam: no decay data for {nuclide_name(*s)}; treated as stable "
                            "(chain truncated)")
            children[s] = []
            continue
        kids = []
        for d, br in rec[1]:
            if d is None or br <= 0:
                continue
            if decay.get(d) is None and d[2] > 0 and decay.get((d[0], d[1], 0)) is not None:
                d = (d[0], d[1], 0)
            kids.append(d)
            todo.append(d)
        children[s] = kids
    if di not in seen:
        return None
    parents: dict[StateKey, list[StateKey]] = {}
    for s, kids in children.items():
        for d in kids:
            parents.setdefault(d, []).append(s)
    anc, todo = {di}, [di]                          # reverse closure: everything that leads to the DI
    while todo:
        for p in parents.get(todo.pop(), []):
            if p not in anc:
                anc.add(p)
                todo.append(p)
    return [target] + sorted(anc - {target})


def decay_matrix(nodes: list[StateKey], decay: DecayLibrary) -> np.ndarray:
    """D[i, j] = lambda_j * BR(j -> i) for j != i and D[j, j] = -lambda_j [s^-1].
    Branches leaving the network are losses (already in the diagonal)."""
    index = {s: k for k, s in enumerate(nodes)}
    D = np.zeros((len(nodes), len(nodes)))
    for j, s in enumerate(nodes):
        rec = decay.get(s)
        if rec is None:
            continue
        lam, branches = rec
        D[j, j] -= lam
        for d, br in branches:
            if d is not None and decay.get(d) is None and d[2] > 0:
                d = (d[0], d[1], 0)
            if d in index and d != s:
                D[index[d], j] += lam * br
    return D


def production_states(st: State, target: StateKey) -> tuple[dict[StateKey, float], float]:
    """Production rates P_i [atoms/s] into each (Z, A, I) state and target burnup P_T
    [atoms/s], from one ISOTOPIA run.

    ISOTOPIA gives R_i [s^-1] per target atom (reactionrates.f90) for the N_0 atoms in
    the traversed length Leff, so P_i = N_0 R_i is the physical production rate. A
    nuclide with isomer-resolved cross sections has rates for is = 0, 1, 2 and a total
    (is = -1); only the resolved states are used, so the total is not counted twice.
    Inelastic scattering back to the target's own ground state changes nothing and is
    skipped.
    """
    N0 = float(st.N_0)
    P: dict[StateKey, float] = {}
    for (z, a, is_), R in st.reaction_rate.items():
        if (z, a) == (0, 0) or float(R) <= 0 or not st.rpexist[(z, a, is_)]:
            continue
        resolved = st.Nisomer[(z, a)] >= 0
        if (resolved and is_ < 0) or (not resolved and is_ != -1):
            continue
        s = (z, a, max(is_, 0))
        if s != target:
            P[s] = N0 * float(R)
    return P, N0 * float(st.reaction_rate[(0, 0, -1)])


@dataclass
class StepProblem:
    """Inventory of one target's DI network under a step beam.

    energies: grid energies [MeV] (sorted) with production tables P [atoms/s], shape
    (n_energies, n_nodes), and burnup P_T [atoms/s]; D: decay matrix [s^-1]; N_T0:
    physical target atoms; t_irr, t_cool [s]. Node 0 is the target.
    """
    energies: np.ndarray
    P: np.ndarray
    P_T: np.ndarray
    D: np.ndarray
    N_T0: float
    t_irr: float
    t_cool: float
    di_index: int

    def __post_init__(self):
        from scipy.linalg import expm
        self._expm = expm
        self._cool = expm(self.D * self.t_cool) if self.t_cool > 0 else np.eye(len(self.D))

    def _rates(self, E: float) -> tuple[np.ndarray, float]:
        """Production and burnup at energy E [MeV], linear between grid energies (as ISOTOPIA's pol1)."""
        k = int(np.clip(np.searchsorted(self.energies, E) - 1, 0, len(self.energies) - 2))
        w = (E - self.energies[k]) / (self.energies[k + 1] - self.energies[k])
        w = min(max(w, 0.0), 1.0)
        return (1 - w) * self.P[k] + w * self.P[k + 1], (1 - w) * self.P_T[k] + w * self.P_T[k + 1]

    def evolve(self, step_energies, durations) -> tuple[np.ndarray, np.ndarray]:
        """Atoms of every node at EOI and at end of cooling for a step beam.

        Within a step: dN/dt = M N with M = D plus the target column, M[i, 0] = P_i/N_T0
        (production in proportion to the remaining target) and M[0, 0] -= P_T/N_T0 (burnup,
        spread uniformly over the target).
        """
        N = np.zeros(len(self.D))
        N[0] = self.N_T0
        for E, dt in zip(step_energies, durations):
            if dt <= 0:
                continue
            P, P_T = self._rates(E)
            M = self.D.copy()
            M[:, 0] += P / self.N_T0
            M[0, 0] -= P_T / self.N_T0
            N = self._expm(M * dt) @ N
        return N, self._cool @ N


def optimise_steps(prob: StepProblem, n_steps: int, metric: str, seed: int = 0) -> dict:
    """Best step energies and switch times for the DI atoms at EOC or EOI.

    Variables: n energies in [E_min, E_max] of the grid, and n - 1 switch points
    u in [0, 1] (sorted, times = u * t_irr), so the total irradiation time is fixed.
    scipy's differential_evolution (Storn & Price, J. Global Optim. 11 (1997) 341)
    searches them; its population includes the best constant beam, so the result is
    never worse than a constant energy.
    """
    from scipy.optimize import differential_evolution
    pick = 1 if metric == "eoc" else 0
    E_lo, E_hi = float(prob.energies[0]), float(prob.energies[-1])

    def schedule(x):
        cuts = np.concatenate([[0.0], np.sort(x[n_steps:]), [1.0]]) * prob.t_irr
        return np.asarray(x[:n_steps]), np.diff(cuts), cuts[:-1]

    def atoms(x):
        E, dt, _ = schedule(x)
        return prob.evolve(E, dt)[pick][prob.di_index]

    const = [prob.evolve([E], [prob.t_irr])[pick][prob.di_index] for E in prob.energies]
    k0 = int(np.argmax(const))
    best_const = {"E": float(prob.energies[k0]), "atoms": float(const[k0])}
    if n_steps == 1:
        return {"constant": best_const, "energies": [best_const["E"]], "durations": [prob.t_irr],
                "starts": [0.0], "atoms": best_const["atoms"]}
    x0 = [best_const["E"]] * n_steps + list(np.linspace(0, 1, n_steps + 1)[1:-1])
    res = differential_evolution(lambda x: -atoms(x), [(E_lo, E_hi)] * n_steps + [(0.0, 1.0)] * (n_steps - 1),
                                 x0=x0, seed=seed, tol=1e-10, atol=0, maxiter=300, polish=True)
    x = res.x if -res.fun >= best_const["atoms"] else np.asarray(x0)
    E, dt, t0 = schedule(x)
    return {"constant": best_const, "energies": E.tolist(), "durations": dt.tolist(), "starts": t0.tolist(),
            "atoms": float(atoms(x))}


def run_rates(spec: RunSpec) -> dict:
    """One ISOTOPIA run at one energy, returning production rates by state (picklable); never raises."""
    t = spec.target
    out = {"projectile": PARSYM[spec.k0], "target": t.name, "Ebeam": spec.ebeam, "Eback": spec.eback,
           "stopping": None, "status": "ok", "note": ""}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            st = run(spec.lines, out=io.StringIO(), outdir=tmp)
    except (IsotopiaError, OSError, ValueError) as exc:
        out.update(status="error", note=" ".join(str(exc).split()))
        return out
    P, P_T = production_states(st, (t.Z, t.A, 0))
    out.update(P=P, P_T=P_T, N0=float(st.N_0), Leff=float(st.Leff), t_irr=float(st.Tir), t_cool=float(st.Tco),
               Eback=float(st.Eback), note=stop_note(st),
               stopping=st.stopping_model if spec.k0 > 1 else None)
    return out


def run_step_search(args, di: StateKey, di_name: str, M_di: float, specs: list[RunSpec],
                    exclude: set[Nuclide], warnings: list[str], path: str) -> str:
    """Rates for every target on the energy grid, then the best N-step beam per target."""
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(run_rates, specs, chunksize=8))
    else:
        results = [run_rates(s) for s in specs]
    by_target: dict[tuple[str, str], list[dict]] = {}
    for spec, r in zip(specs, results):
        if r["status"] != "ok":
            warnings.append(f"{PARSYM[spec.k0]}+{r['target']} at {r['Ebeam']} MeV: {r['note']}")
            continue
        if r["note"]:
            warnings.append(f"{PARSYM[spec.k0]}+{r['target']} at {r['Ebeam']} MeV: {r['note']}")
        by_target.setdefault((PARSYM[spec.k0], r["target"]), []).append(r)
    warnings.extend(stopping_warnings(results))
    decay = DecayLibrary(path)
    rarity = RarityIndex(args.rarity) if args.rarity else None
    r_di = rarity.lookup(*di, args.rarity_basis) if rarity else None
    if r_di:
        warnings.extend(f"rarity: {f}" for f in r_di[1])
    pick = "eoc" if args.metric == "eoc" else "eoi"
    g_per_atom = M_di / float(AVOGADRO)
    results_rows = []
    for (p, tname), runs in sorted(by_target.items()):
        runs.sort(key=lambda r: r["Ebeam"])
        if len(runs) < 2:
            warnings.append(f"step beam: {p}+{tname} has fewer than 2 usable energies; skipped")
            continue
        Zt, At, _ = parse_nuclide(tname)
        target = (Zt, At, 0)
        produced = set().union(*(r["P"].keys() for r in runs))
        nodes = build_network(target, di, produced, decay, warnings)
        row = {"projectile": p, "target": tname, "status": "ok", "note": ""}
        if nodes is None:
            row.update(status="not_produced", note="DI not reachable from any produced nuclide")
            results_rows.append(row)
            continue
        hit = next((listed(*s, exclude) for s in nodes[1:] if s != di and listed(*s, exclude)), None)
        if hit:
            row.update(status="excluded", note=f"chain member {nuclide_name(*hit)} on exclusion list")
            results_rows.append(row)
            continue
        idx = {s: k for k, s in enumerate(nodes)}
        # Physical target atoms: ISOTOPIA's N_0 covers only the traversed length Leff;
        # the whole target is the --thickness, the same at every energy
        r0 = max(runs, key=lambda r: r["Leff"])
        N_T0 = r0["N0"] * args.thickness / r0["Leff"]
        P = np.array([[r["P"].get(s, 0.0) if s != target else 0.0 for s in nodes] for r in runs])
        prob = StepProblem(np.array([r["Ebeam"] for r in runs]), P, np.array([r["P_T"] for r in runs]),
                           decay_matrix(nodes, decay), N_T0, runs[0]["t_irr"],
                           runs[0]["t_cool"], idx[di])
        best = optimise_steps(prob, args.steps, pick, seed=args.seed)
        weight = 1.0
        if rarity:
            r_t, flags = rarity.lookup(*target, args.rarity_basis)
            warnings.extend(f"rarity: {f}" for f in flags)
            delta, weight = rarity_weight(r_t, r_di[0])
            if delta < 0:
                warnings.append(f"rarity: target {tname} is rarer than the DI; kept with weight {weight:.3f}")
        row.update(chain=", ".join(nuclide_name(*s) for s in nodes[1:]),
                   m_const_g=best["constant"]["atoms"] * g_per_atom, E_const=best["constant"]["E"],
                   m_step_g=best["atoms"] * g_per_atom, schedule=best, weight=weight,
                   score=best["atoms"] * g_per_atom * weight)
        results_rows.append(row)
    warnings.extend(f"decay data: {n}" for n in dict.fromkeys(decay.notes))
    return write_step_outputs(args, di_name, results_rows, warnings)


def write_step_outputs(args, di_name: str, rows: list[dict], warnings: list[str]) -> str:
    """steps.csv (one line per step of each target's best schedule) and summary_steps.txt."""
    outdir = Path(args.out) if args.out else ROOT / "runs" / "optimal_route" / f"{di_name}_steps{args.steps}"
    outdir.mkdir(parents=True, exist_ok=True)
    ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: r["score"], reverse=True)
    with open(outdir / "steps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["projectile", "target", "step", "E_MeV", "start_s", "duration_s",
                    "m_step_g", "m_const_g", "E_const_MeV", "rarity_weight", "score", "chain"])
        for r in ok:
            s = r["schedule"]
            for k, (E, t0, dt) in enumerate(zip(s["energies"], s["starts"], s["durations"]), 1):
                w.writerow([r["projectile"], r["target"], k, f"{E:.4f}", f"{t0:.6g}", f"{dt:.6g}",
                            f"{r['m_step_g']:.6e}", f"{r['m_const_g']:.6e}", f"{r['E_const']:.4f}",
                            f"{r['weight']:.4f}", f"{r['score']:.6e}", r["chain"]])
    when = "end of cooling" if args.metric == "eoc" else "end of irradiation"
    settings = (f"Projectiles: {' '.join(args.projectile)}   grid: {args.emin}-{args.emax} MeV, "
                f"step {args.de} MeV   thickness: {args.thickness} cm   Ibeam: {args.ibeam} mA   "
                f"Tirrad: {args.tirrad}   Tcool: {args.tcool}")
    out = [f"Step-beam optimisation for {di_name}: {args.steps} energy steps", settings,
           f"Objective: DI mass at {when}" + (f" x rarity weight ({args.rarity_basis})" if args.rarity else ""),
           "", STEP_ASSUMPTIONS]
    for r in ok:
        s = r["schedule"]
        gain = r["m_step_g"] / r["m_const_g"] - 1 if r["m_const_g"] > 0 else 0.0
        out.append(f"{r['projectile']}+{r['target']}: m = {r['m_step_g']:.4e} g with steps vs "
                   f"{r['m_const_g']:.4e} g at the best constant {r['E_const']:.3g} MeV ({gain:+.2%})"
                   + (f", rarity weight {r['weight']:.3f}" if args.rarity else ""))
        for k, (E, t0, dt) in enumerate(zip(s["energies"], s["starts"], s["durations"]), 1):
            out.append(f"    step {k}: {E:8.3f} MeV  from t = {t0 / 3600:9.3f} h  for {dt / 3600:9.3f} h")
        out.append(f"    chain feeding the DI: {r['chain']}")
    if ok:
        r = ok[0]
        out += ["", (f"OPTIMAL: {r['target']}({r['projectile']},x){di_name} with {args.steps} steps, "
                     f"m = {r['m_step_g']:.4e} g")]
    else:
        out += ["", "OPTIMAL: none - no route produced the DI"]
    for r in rows:
        if r["status"] != "ok":
            out.append(f"{r['projectile']}+{r['target']}: {r['status']} ({r['note']})")
    out += ["", "Warnings"] + ([f"- {w}" for w in dict.fromkeys(warnings)] or ["- none"])
    text = "\n".join(out) + "\n"
    (outdir / "summary_steps.txt").write_text(text, encoding="utf-8")
    return text


STEP_ASSUMPTIONS = """\
Assumptions and limitations (step beam)
- NEW PHYSICS: piecewise-constant beam energy solved exactly per step with the
  matrix exponential; full decay chains (all modes, branching ratios, daughter
  isomers) from files/decay. This differs from the constant-energy search, which
  uses ISOTOPIA's one-step beta feeding, so the two can disagree for DIs fed by
  decay (the step-beam numbers are the more physical ones).
- Reaction rates from ISOTOPIA at the grid energies, linear in between; Eback of
  every step solved by ISOTOPIA from the fixed target thickness; beam current constant.
- Total irradiation time fixed; only switch times and step energies vary.
- Products are not hit again by the beam (no double hitting); target burnup is
  spread uniformly over the target.
- Only nuclides that can decay into the DI are tracked; states without decay data
  are treated as stable (chain truncated, warned).
- The optimiser starts from the best constant beam, so steps never do worse; for a
  DI made only directly, the best schedule is usually a constant beam.
"""


def library_emin(crosspath: str, k0: int, t: Target) -> float | None:
    """Lowest energy in the target's non-elastic file (MT005, or MT003 for neutrons), MeV."""
    mt = "MT003" if k0 == 1 else "MT005"
    nucname = f"{nuc(t.Z).strip()}{t.A:03d}"
    f = Path(crosspath) / PARSYM[k0] / nucname / "iaea.2026" / "tables" / "xs" / \
        f"{PARSYM[k0]}-{nucname}-{mt}.iaea.2026"
    if not f.exists():
        return None
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() and not line.startswith("#"):
            return float(line.split()[0])
    return None


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

CSV_FIELDS = ["projectile", "target", "stable", "Ebeam_MeV", "Eback_MeV", "Leff_cm", "stopping",
              "m_rate", "m_rate_unit", "m_EOI_g", "m_EOC_g",
              "R_target", "R_DI", "delta_R", "rarity_weight", "score", "rarity_flags", "status", "note"]

# Ranking options: --metric name -> (CSV column, description)
METRICS = {
    "eoc": ("m_EOC_g", "DI mass at end of cooling [g]"),
    "eoi": ("m_EOI_g", "DI mass at end of irradiation [g]"),
    "rate": ("m_rate", "initial mass production rate [g/(mA h); g/(g h) for n and g]"),
}

ASSUMPTIONS = """\
Assumptions and limitations
- Reported for every route: DI mass at EOI and at end of cooling (last point of
  ISOTOPIA's time grid), and the initial mass production rate. Ranking uses the
  chosen metric. Masses are m = N * M / N_A, with N the atom count ISOTOPIA
  computes and M = AWR * m_n read from the DI's record in files/decay/zZZZ.
- Stable and radioactive DIs are treated alike. Masses are of the DI alone: for a
  stable DI, other isotopes of the same element (including the target, if it is
  the same element) cannot be separated chemically, and isotopic purity is not
  reported yet.
- The mass rate is per mA h of beam for charged particles; for neutrons and
  photons ISOTOPIA normalises per gram of target, so rates are not comparable
  across those groups.
- Targets: mass number A_DI-5..A_DI+5, ground state, with a cross-section library;
  only Z and A inside ISOTOPIA's residual window (Zcomp+1 .. Zcomp-Zdepth,
  Acomp .. Acomp-Adepth); lighter targets need A + A_projectile >= A_DI.
- Charged particles: the thickness is passed to ISOTOPIA ('thickness' keyword),
  which solves Eback with its own stopping-power model: SRIM tables where they
  exist (p, d, t, h, a in H to U, Z = 1-92), otherwise Bethe-Bloch
  (isotopia_py.rates._setup_target), and the 25 degC density table. The model
  used is in the 'stopping' column, and every target on Bethe-Bloch is listed
  under Warnings. Eback_MeV and Leff_cm are ISOTOPIA's values. If the beam stops
  in the target, SRIM runs set Eback to the lowest table energy and irradiate
  only the depth the beam reaches (Leff_cm); Bethe-Bloch runs fail instead,
  because the formula is not valid at those energies. Both are listed under
  Warnings.
- Decay physics is the Phase 1 port of equations.f90, including its known defects
  (claude/equations_f90_findings.md): beta- parents do NOT feed daughters during
  cooling, no branching ratios, double-counted parent feeding, no isomeric
  transition. End-of-cooling masses of DIs fed by beta- decay (including stable
  DIs made mostly by decay) are therefore underestimated; compare m_EOI_g.
- Exclusion checks the target and the one-step decay precursors ISOTOPIA models;
  longer chains are not tracked by the code.
- Radiotoxicity and processing time are not part of the ranking yet.
- Rarity (only with --rarity): index from the team workbook, U.S. market prices
  (log-normalised over the whole sheet) or U.S. obtainability; region selection is
  not possible because the workbook holds one value per element. Element prices
  stand in for isotopes without their own row, so enriched targets (e.g. Mo-100)
  are scored as the natural element and their rarity is underestimated.
"""


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6e}" if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-2) else f"{v:.6f}"
    return str(v)


def best_per_route(rows: list[dict], col: str) -> dict[tuple[str, str], dict]:
    """Row with the largest `col` value for each (projectile, target) route."""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["projectile"], r["target"])
        if k not in best or r[col] > best[k][col]:
            best[k] = r
    return best


def _route(r: dict, di_name: str) -> str:
    """One-line route description, e.g. 'Mo100(p,x)Tc99_m1 at Ebeam = 20 MeV'."""
    e = "reactor spectrum" if r["Ebeam_MeV"] is None else f"Ebeam = {r['Ebeam_MeV']:.4g} MeV"
    return f"{r['target']}({r['projectile']},x){di_name} at {e}"


def write_outputs(outdir: Path, di_name: str, rows: list[dict], warnings: list[str], args) -> str:
    """Write scan.csv and summary.txt; return the summary text."""
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "scan.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in CSV_FIELDS})

    col, desc = METRICS[args.metric]
    ok = [r for r in rows if r["status"] == "ok" and r[col] is not None]
    settings = (f"Projectiles: {' '.join(args.projectile)}   grid: {args.emin}-{args.emax} MeV, "
                f"step {args.de} MeV   thickness: {args.thickness} cm")
    beam = (f"Ibeam: {args.ibeam} mA   Tirrad: {args.tirrad}   Tcool: {args.tcool}   "
            f"exclusion list: {args.exclude or 'none'}")
    out = [f"Optimal route search for {di_name}", "", settings, beam,
           f"Ranked by: {desc}", "", ASSUMPTIONS]

    # Best energy per (projectile, target) under the chosen metric, then rank the routes.
    # The rarity weight is the same for every energy of a route, so it changes the
    # order of routes but never which energy is best for a route.
    best = best_per_route(ok, col)
    key = "score" if args.rarity else col
    ranked = sorted(best.values(), key=lambda r: r[key], reverse=True)
    if args.rarity:
        desc = f"{desc} x rarity weight ({args.rarity_basis})"
    out.append(f"Best energy per route (ranked by {desc})")
    out.append(f"{'proj':>4} {'target':>8} {'stable':>6} {'Ebeam':>9} {'Eback':>9} "
               f"{'rate':>11} {'rate unit':>10} {'m_EOI [g]':>11} {'m_EOC [g]':>11}"
               + (f" {'R_tgt':>6} {'dR':>6} {'weight':>6} {'score':>11}" if args.rarity else ""))
    for r in ranked:
        line = (f"{r['projectile']:>4} {r['target']:>8} {'yes' if r['stable'] else 'no':>6} "
                f"{_fmt(r['Ebeam_MeV']):>9} {_fmt(r['Eback_MeV']):>9} "
                f"{r['m_rate']:11.4e} {r['m_rate_unit']:>10} {r['m_EOI_g']:11.4e} {r['m_EOC_g']:11.4e}")
        if args.rarity:
            line += (f" {r['R_target']:6.3f} {r['delta_R']:+6.3f} {r['rarity_weight']:6.3f} {r['score']:11.4e}"
                     + ("  <- target rarer than DI" if r["delta_R"] < 0 else ""))
        out.append(line)
    out.append("")
    if ranked and ranked[0][key] > 0:
        r = ranked[0]
        extra = f", rarity weight = {r['rarity_weight']:.3f}" if args.rarity else ""
        out.append(f"OPTIMAL ({args.metric}{' + rarity' if args.rarity else ''}): {_route(r, di_name)}, "
                   f"m_EOC = {r['m_EOC_g']:.4e} g, rate = {r['m_rate']:.4e} {r['m_rate_unit']}{extra}")
    else:
        out.append("OPTIMAL: none - no route produced the DI")

    # Cross-check: the best route under every metric. Disagreement means half-life
    # and timing, not the cross section alone, are deciding the choice.
    out.append("Best route under each metric:")
    for name, (c, _) in METRICS.items():
        cands = [r for r in ok if r[c] is not None and r[c] > 0]
        top = max(cands, key=lambda r: r[c]) if cands else None
        out.append(f"  {name:>5}: {_route(top, di_name) if top else 'none'}")
    out.append("")

    # Counts of non-ok runs, then warnings
    for status in ("excluded", "not_produced", "no_data", "error"):
        n = sum(r["status"] == status for r in rows)
        if n:
            out.append(f"{n} runs with status '{status}' (see scan.csv)")
    never = sorted({(r["projectile"], r["target"]) for r in rows} - set(best))
    if never:
        out.append("Routes with no usable result at any energy: "
                   + ", ".join(f"{t}({p},x)" for p, t in never))
    out.append("")
    out.append("Warnings")
    if warnings:
        out.extend(f"- {w}" for w in warnings)
    else:
        out.append("- none")
    text = "\n".join(out) + "\n"
    (outdir / "summary.txt").write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("di", help="desired isotope, e.g. Tc99_m1, Cu64, Tc99_g")
    ap.add_argument("-p", "--projectile", nargs="+", required=True, choices=PARSYM,
                    help="projectile(s): g n p d t h a")
    ap.add_argument("--emin", type=float, default=0.001, help="lowest beam energy [MeV] (default 0.001)")
    ap.add_argument("--emax", type=float, default=200.0, help="highest beam energy [MeV] (default 200)")
    ap.add_argument("--de", type=float, required=True, help="energy step [MeV]")
    ap.add_argument("--thickness", type=float, default=0.05,
                    help="target thickness [cm] (default 0.05, the ISOTOPIA default)")
    ap.add_argument("--ibeam", type=float, default=1.0, help="beam current [mA] (default 1)")
    ap.add_argument("--tirrad", default="1 d", help="irradiation time, ISOTOPIA syntax (default '1 d')")
    ap.add_argument("--tcool", default="1 d", help="cooling time, ISOTOPIA syntax (default '1 d')")
    ap.add_argument("--flux", type=float, default=None, help="neutron flux [cm^-2 s^-1] (n only)")
    ap.add_argument("--exclude", default=None, help="text file of nuclides that exclude a route")
    ap.add_argument("--include-radioactive", action="store_true",
                    help="also use non-naturally-occurring (radioactive) targets")
    ap.add_argument("--metric", choices=list(METRICS), default="eoc",
                    help="ranking: eoc = DI mass at end of cooling (default), eoi = DI mass at "
                         "end of irradiation, rate = initial mass production rate")
    ap.add_argument("--rarity", default=None, metavar="WORKBOOK",
                    help="weight the ranking by rarity from the team price/obtainability workbook (.xlsx)")
    ap.add_argument("--rarity-basis", choices=RARITY_BASES, default="price",
                    help="rarity basis (default price); the other basis is the fallback")
    ap.add_argument("--steps", type=int, default=None, metavar="N",
                    help="optimise a step-function beam with N energy steps (charged particles only); "
                         "uses full decay chains and a time-dependent solver")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the step-beam optimiser (default 0)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--out", default=None, help="output directory (default runs/optimal_route/<DI>)")
    args = ap.parse_args(argv)
    if not 0 < args.emin < args.emax <= 200.0:
        ap.error("need 0 < emin < emax <= 200 MeV (library upper limit)")
    if args.de <= 0 or args.thickness <= 0:
        ap.error("--de and --thickness must be positive")
    if args.steps is not None and (args.steps < 1 or args.metric == "rate"):
        ap.error("--steps needs N >= 1 and --metric eoc or eoi")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    di = parse_nuclide(args.di)
    if di[2] is None:
        di = (di[0], di[1], 0)                 # bare DI name means the ground state
    di_name = nuclide_name(*di)
    exclude = read_nuclide_list(args.exclude) if args.exclude else set()
    warnings: list[str] = []
    if listed(*di, exclude):
        warnings.append(f"the DI {di_name} itself is on the exclusion list")
    groups = {p in "pdtha" for p in args.projectile}
    if args.metric == "rate" and len(groups) > 1:
        warnings.append("--metric rate mixes g/(mA h) (charged particles) with g/(g h) "
                        "(n, g); the ranking across those groups is not meaningful")

    paths = State()
    machine(paths)                             # resolves ISOTOPIA_DIR / ISOTOPIA_LIBS
    try:
        M_di = molar_mass(paths.path, di[0], di[1])   # g/mol, from the decay library
    except LookupError as exc:
        print(f"error: cannot convert atoms to grams for {di_name}: {exc}", file=sys.stderr)
        return 1
    grid = energy_grid(args.emin, args.emax, args.de)
    specs: list[RunSpec] = []
    rows: list[dict] = []
    if args.steps is not None:
        dropped = [p for p in args.projectile if p in "gn"]
        if dropped:
            warnings.append(f"step beam: {' '.join(dropped)} skipped (photon spectra exist only at whole "
                            "MeV and the reactor spectrum has no beam energy)")
        args.projectile = [p for p in args.projectile if p not in "gn"]
    for p in dict.fromkeys(args.projectile):
        k0 = PARSYM.index(p)
        targets = candidate_targets(di, k0, paths.path, paths.crosspath,
                                    args.include_radioactive, warnings)
        if not targets:
            warnings.append(f"{p}: no candidate targets with A = {di[1] - MASS_WINDOW}..{di[1] + MASS_WINDOW} "
                            f"in {paths.crosspath}{p}/")
        for t in targets:
            emin_lib = library_emin(paths.crosspath, k0, t)
            if emin_lib is not None and k0 != 1 and args.emin < emin_lib:
                warnings.append(f"{p}+{t.name}: library cross sections start at {emin_lib:g} MeV; "
                                "lower beam energies give zero reaction rates")
        s, skipped = build_specs(di, k0, targets, grid, args, exclude, warnings)
        for spec in s:
            spec.molar_mass = M_di
        specs += s
        rows += skipped

    print(f"{di_name}: {len(specs)} runs on {len({(s.k0, s.target) for s in specs})} routes", file=sys.stderr)
    if args.steps is not None:
        print(run_step_search(args, di, di_name, M_di, specs, exclude, warnings, paths.path))
        return 0
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows += list(pool.map(run_one, specs, chunksize=8))
    else:
        rows += [run_one(s) for s in specs]
    for r in rows:
        if r["status"] == "error" or r.get("note", "").startswith(BEAM_STOPS):
            warnings.append(f"{r['projectile']}+{r['target']} at {r.get('Ebeam_MeV')} MeV: {r['note']}")
    warnings.extend(stopping_warnings(rows))
    if args.rarity:
        apply_rarity(rows, di, RarityIndex(args.rarity), args.rarity_basis, METRICS[args.metric][0], warnings)

    outdir = Path(args.out) if args.out else ROOT / "runs" / "optimal_route" / di_name
    print(write_outputs(outdir, di_name, rows, warnings, args))
    print(f"results in {outdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
