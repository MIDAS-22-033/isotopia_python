"""Tests for the price / obtainability lookup (needs isotopia_py/data/element_economics.csv)."""
from __future__ import annotations

import pytest

from isotopia_py import economics as E


def test_natural_element_all_regions():
    for region in E.REGIONS:
        mo = E.lookup(42, region=region)
        assert mo.symbol == "Mo" and mo.exact
        assert mo.price_usd_per_g and mo.price_usd_per_g > 0
        assert 0.0 <= mo.price_index <= 1.0
        assert mo.obtainability is None or 0.0 <= mo.obtainability <= 1.0


def test_isotope_row_and_fallback():
    u235 = E.lookup(92, 235, 0)
    assert u235.exact and u235.price_usd_per_g > E.lookup(92).price_usd_per_g
    mo100 = E.lookup(42, 100, 0)          # no Mo-100 row: falls back to natural Mo
    assert not mo100.exact and mo100.price_usd_per_g == E.lookup(42).price_usd_per_g


def test_isomer_row():
    tc99m = E.lookup(43, 99, 1)
    assert tc99m.exact and tc99m.price_usd_per_g > 1e6


def test_cost_and_bad_region():
    au = E.lookup(79)
    assert au.cost_usd(2.0) == pytest.approx(2.0 * au.price_usd_per_g)
    with pytest.raises(ValueError):
        E.lookup(79, region="mars")
