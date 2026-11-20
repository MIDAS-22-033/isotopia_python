"""Handling-hazard indices for targets and products.

A proxy for how expensive / dangerous it is to handle an irradiated target or to
separate a product. Two data files, exported from Element_Hazard_Levels.xlsx
(rebuilt by element-prices/build_hazard_index.py):

* ``element_hazard.csv`` -- one row per element (and a few isotopes), keyed like
  ``element_economics.csv``: ``(Z, A, isomer)`` with ``A = 0, isomer = -1`` for
  the natural element. Gives a 0-10 ``hazard_score`` combining a chemical score
  (reactivity / flammability weighted high, ingestion / chronic weighted low)
  and a radiological score for the element's representative isotope.
* ``nuclide_radiological_index.csv`` -- every radioactive nuclide in JEFF-3.1.1,
  keyed ``(Z, A, isomer)`` with isomer 0 = ground state, 1 = first isomer.
  The 0-10 score reflects radiation *quality* per decay (alpha, spontaneous-
  fission neutrons, gamma and beta energy, short-lived daughters included),
  not activity: isotopia supplies the activity.

Scores run 0-10; ``*_index`` values are score / 10 (0-1, like the price indices).
Classes: 0 Negligible, 1 Very low, 2 Low, 3 Moderate, 4 Elevated, 5 High,
6 Very high, 7 Severe.

Files are looked up in ``$ISOTOPIA_HAZARD`` / ``$ISOTOPIA_NUCLIDE_HAZARD`` if
set, otherwise in ``isotopia_py/data/``.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DATA = Path(__file__).resolve().parent / "data"
_DEFAULT_ELEMENT = _DATA / "element_hazard.csv"
_DEFAULT_NUCLIDE = _DATA / "nuclide_radiological_index.csv"

CLASSES = ((9.0, "Severe"), (8.0, "Very high"), (6.5, "High"), (5.0, "Elevated"),
           (3.5, "Moderate"), (2.0, "Low"), (0.5, "Very low"), (float("-inf"), "Negligible"))


def hazard_class(score: float) -> str:
    """'<level> – <name>' for a 0-10 score, using the workbook's class bounds."""
    score = round(score, 2)
    for i, (lo, name) in enumerate(CLASSES):
        if score >= lo:
            return f"{len(CLASSES) - 1 - i} – {name}"
    raise AssertionError


def combine(radiological: float, chemical: float) -> float:
    """Workbook rule: max + 0.2 × min, capped at 10."""
    return min(10.0, max(radiological, chemical) + 0.2 * min(radiological, chemical))


@dataclass(frozen=True)
class Hazard:
    """Handling hazard of a target (element row of the workbook)."""

    Z: int
    A: int
    isomer: int
    symbol: str
    form: str
    hazard_score: float
    hazard_class: str
    radiological_score: float
    chemical_score: Optional[float]  # None: no GHS data (synthetic / radioactive-only elements)
    exact: bool  # False when an isotope wasn't listed and the natural-element row was used

    @property
    def hazard_index(self) -> float:
        return self.hazard_score / 10

    @property
    def radiological_index(self) -> float:
        return self.radiological_score / 10

    @property
    def chemical_index(self) -> Optional[float]:
        return None if self.chemical_score is None else self.chemical_score / 10


@dataclass(frozen=True)
class NuclideHazard:
    """Radiological score of one nuclide, plus the combined score for handling it as a product."""

    Z: int
    A: int
    isomer: int
    nuclide: str
    half_life_s: Optional[float]  # None for stable / not in JEFF-3.1.1
    radiological_score: float
    alpha_score: float
    sf_neutron_score: float
    gamma_score: float
    beta_score: float
    low_sa_penalty: float
    chemical_score: Optional[float]  # of the element, from element_hazard.csv
    product_score: float             # combine(radiological, chemical or 0)

    @property
    def radioactive(self) -> bool:
        return self.half_life_s is not None

    @property
    def radiological_index(self) -> float:
        return self.radiological_score / 10

    @property
    def product_index(self) -> float:
        return self.product_score / 10

    @property
    def product_class(self) -> str:
        return hazard_class(self.product_score)


def _num(s) -> Optional[float]:
    return None if s in ("", None) else float(s)


def _path(path, env: str, default: Path) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ[env]) if os.environ.get(env) else default


@lru_cache(maxsize=4)
def load(path: Optional[str] = None) -> dict[tuple[int, int, int], dict]:
    """Element table ``{(Z, A, isomer): row}``. Cached per path."""
    table: dict[tuple[int, int, int], dict] = {}
    with open(_path(path, "ISOTOPIA_HAZARD", _DEFAULT_ELEMENT), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table.setdefault((int(row["z"]), int(row["a"]), int(row["isomer"])), row)
    return table


@lru_cache(maxsize=4)
def load_nuclides(path: Optional[str] = None) -> dict[tuple[int, int, int], dict]:
    """Nuclide table ``{(Z, A, isomer): row}``. Cached per path."""
    table: dict[tuple[int, int, int], dict] = {}
    with open(_path(path, "ISOTOPIA_NUCLIDE_HAZARD", _DEFAULT_NUCLIDE), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table[(int(row["z"]), int(row["a"]), int(row["isomer"]))] = row
    return table


def _element_row(table, Z: int) -> Optional[dict]:
    """Natural-element row, or the first row for Z (Pm, Po, Pu... are listed by isotope only)."""
    row = table.get((Z, 0, -1))
    if row is None:
        rows = [r for k, r in table.items() if k[0] == Z]
        row = rows[0] if rows else None
    return row


def lookup(Z: int, A: int = 0, isomer: int = -1, path: Optional[str] = None) -> Hazard:
    """Handling hazard of target ``(Z, A, isomer)``.

    ``A = 0`` (or an isotope without its own row) returns the element row.
    Raises KeyError if the element is not in the table.
    """
    table = load(None if path is None else str(path))
    exact = True
    row = table.get((Z, A, isomer)) or (table.get((Z, A, 0)) if isomer == -1 and A else None)
    if row is None:
        row = _element_row(table, Z)
        exact = A == 0
        if row is None:
            raise KeyError(f"no hazard data for Z={Z}")
    return Hazard(
        Z=Z, A=A, isomer=isomer, symbol=row["symbol"], form=row["form"],
        hazard_score=float(row["hazard_score"]), hazard_class=row["hazard_class"],
        radiological_score=float(row["radiological_score"]),
        chemical_score=_num(row["chemical_score"]), exact=exact,
    )


def nuclide(Z: int, A: int, isomer: int = 0, path: Optional[str] = None,
            element_path: Optional[str] = None) -> NuclideHazard:
    """Radiological and product score of nuclide ``(Z, A, isomer)``.

    ``isomer = -1`` (isotopia's "total") is read as the ground state. Stable
    nuclides and nuclides missing from JEFF-3.1.1 get a radiological score of 0.
    """
    iso = 0 if isomer < 0 else isomer
    row = load_nuclides(None if path is None else str(path)).get((Z, A, iso))
    erow = _element_row(load(None if element_path is None else str(element_path)), Z)
    chem = _num(erow["chemical_score"]) if erow else None
    if row is None:
        rad = dict(radiological_score=0.0, alpha_score=0.0, sf_neutron_score=0.0, gamma_score=0.0,
                   beta_score=0.0, low_sa_penalty=0.0)
        name, T = f"{erow['symbol'] if erow else Z}-{A}{'m' if iso else ''}", None
    else:
        rad = {k: float(row[k]) for k in ("radiological_score", "alpha_score", "sf_neutron_score",
                                          "gamma_score", "beta_score", "low_sa_penalty")}
        name, T = row["nuclide"], float(row["half_life_s"])
    return NuclideHazard(Z=Z, A=A, isomer=iso, nuclide=name, half_life_s=T, **rad, chemical_score=chem,
                         product_score=combine(rad["radiological_score"], chem or 0.0))


def for_state(st, path: Optional[str] = None) -> Hazard:
    """Handling hazard of the target of a run ``State`` (natural element if Atarget <= 0)."""
    A = st.Atarget if st.Atarget and st.Atarget > 0 else 0
    return lookup(st.Ztarget, A, -1 if A == 0 else 0, path)


def products(st, it: Optional[int] = None) -> list[tuple[NuclideHazard, float]]:
    """``[(NuclideHazard, activity)]`` for every product of a finished run, most active first.

    ``it`` is the time-grid index (default: end of the run, ``st.Ntime``). For each
    (Z, A) the ground/isomer keys are used when isotopia filled them, otherwise the
    total (isomer -1, scored as the ground state), so nothing is counted twice.
    Activities are in isotopia's internal units (multiply by ``st.rfac`` for output units).
    """
    it = st.Ntime if it is None else it
    by_za: dict[tuple[int, int], dict[int, float]] = {}
    for (z, a, iso), act in list(st.activity.items()):
        if st.Yexist.get((z, a, iso), False):
            by_za.setdefault((z, a), {})[iso] = float(act[it])
    out = []
    for (z, a), states in by_za.items():
        split = {i: v for i, v in states.items() if i >= 0}
        for iso, act in (split or {0: states[-1]}).items():
            out.append((nuclide(z, a, iso), act))
    return sorted(out, key=lambda x: -x[1])


def activity_weighted(st, it: Optional[int] = None) -> Optional[float]:
    """Activity-weighted mean product score (0-10) at time index ``it``; None if nothing is active."""
    prods = [(h, a) for h, a in products(st, it) if a > 0]
    total = sum(a for _, a in prods)
    return None if total <= 0 else sum(h.product_score * a for h, a in prods) / total
