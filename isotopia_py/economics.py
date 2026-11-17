"""Market price and supply-obtainability indices for target elements.

Reads ``element_economics.csv`` (exported from the "Pysotopia indices" sheet of
Element_Prices_per_Gram_USD.xlsx). Rows are keyed like the rest of isotopia_py:
``(Z, A, isomer)``, with ``A = 0, isomer = -1`` for the natural element and
explicit rows for a few isotopes whose price differs a lot from the element's
(deuterium, Tc-99m, U-235, U-233, Pu-239/242/244, Cf-249/252, ...).

For each region (``"us"``, ``"eu"``, ``"cn"``) a row gives:

* ``price_usd_per_g``  average market price, US dollars per gram
* ``price_year``       year the price applies to
* ``price_regional``   False when no regional price exists and the U.S. value is used
* ``price_index``      0-1, log scale within that region (0 = cheapest, 1 = most valuable)
* ``price_index_common`` 0-1, one log scale shared by all three regions
* ``obtainability``    0-1, 0 = easiest to obtain in that region, 1 = hardest
                       (None if the element is not ranked)

The data file is looked up in ``$ISOTOPIA_ECONOMICS`` if set, otherwise in
``isotopia_py/data/element_economics.csv``.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

REGIONS = ("us", "eu", "cn")
_DEFAULT = Path(__file__).resolve().parent / "data" / "element_economics.csv"


@dataclass(frozen=True)
class Economics:
    """Price and obtainability of one target in one region."""

    Z: int
    A: int
    isomer: int
    symbol: str
    region: str
    price_usd_per_g: Optional[float]
    price_year: str
    price_regional: bool
    price_index: Optional[float]
    price_index_common: Optional[float]
    obtainability: Optional[float]
    exact: bool  # False when an isotope wasn't listed and the natural-element row was used

    def cost_usd(self, mass_g: float) -> Optional[float]:
        """Material cost of ``mass_g`` grams at this price (None if unpriced)."""
        return None if self.price_usd_per_g is None else self.price_usd_per_g * mass_g


def _num(s: str) -> Optional[float]:
    return None if s in ("", None) else float(s)


def _path(path: Optional[str | os.PathLike]) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("ISOTOPIA_ECONOMICS")
    return Path(env) if env else _DEFAULT


@lru_cache(maxsize=4)
def load(path: Optional[str] = None) -> dict[tuple[int, int, int], dict]:
    """Read the CSV into ``{(Z, A, isomer): row}``. Cached per path."""
    table: dict[tuple[int, int, int], dict] = {}
    with open(_path(path), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["z"]), int(row["a"]), int(row["isomer"]))
            table.setdefault(key, row)  # first row wins (element row before isotope rows)
    return table


def lookup(Z: int, A: int = 0, isomer: int = -1, region: str = "us",
           path: Optional[str] = None) -> Economics:
    """Price and obtainability for target ``(Z, A, isomer)`` in ``region``.

    ``A = 0`` (or an isotope without its own row) returns the natural element.
    Raises KeyError if the element is not in the table.
    """
    region = region.lower()
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    table = load(None if path is None else str(path))
    exact = True
    row = table.get((Z, A, isomer)) or (table.get((Z, A, 0)) if isomer == -1 and A else None)
    if row is None:
        row = table.get((Z, 0, -1))
        exact = A == 0
        if row is None:
            raise KeyError(f"no economics data for Z={Z}")
    r = region
    return Economics(
        Z=Z, A=A, isomer=isomer, symbol=row["symbol"], region=r,
        price_usd_per_g=_num(row[f"price_usd_per_g_{r}"]),
        price_year=row[f"price_year_{r}"],
        price_regional=row[f"price_regional_{r}"] == "True",
        price_index=_num(row[f"price_index_{r}"]),
        price_index_common=_num(row[f"price_index_common_{r}"]),
        obtainability=_num(row[f"obtainability_{r}"]),
        exact=exact,
    )


def for_state(st, region: str = "us", path: Optional[str] = None) -> Economics:
    """Economics of the target of a run ``State`` (natural element if Atarget <= 0)."""
    A = st.Atarget if st.Atarget and st.Atarget > 0 else 0
    return lookup(st.Ztarget, A, -1 if A == 0 else 0, region, path)
