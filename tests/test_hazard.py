"""Tests for the handling-hazard lookup (needs isotopia_py/data/element_hazard.csv and
nuclide_radiological_index.csv; the run test also needs isotopia.libs)."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from isotopia_py import hazard as H

ROOT = Path(__file__).resolve().parents[1]
ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
LIBS = Path(os.environ.get("ISOTOPIA_LIBS") or ISOTOPIA.parent / "isotopia.libs")


def test_element_rows_and_classes():
    cs, au, mo = H.lookup(55), H.lookup(79), H.lookup(42)
    assert cs.exact and cs.hazard_class == "7 – Severe" and cs.chemical_score == 10
    assert au.hazard_score == 0 and au.hazard_class == "0 – Negligible"
    assert 0.0 <= mo.hazard_index <= 1.0
    assert H.lookup(94).chemical_score is None          # Pu: no GHS data
    assert H.lookup(94).form == "Plutonium-239"         # listed by isotope only


def test_isotope_row_and_fallback():
    tc99m = H.lookup(43, 99, 1)
    assert tc99m.exact and tc99m.hazard_class == "2 – Low"
    mo100 = H.lookup(42, 100, 0)                        # no Mo-100 row: natural Mo
    assert not mo100.exact and mo100.hazard_score == H.lookup(42).hazard_score
    with pytest.raises(KeyError):
        H.lookup(200)


def test_nuclide_scores_rank_radiation_quality():
    co60, tc99m, h3 = H.nuclide(27, 60), H.nuclide(43, 99, 1), H.nuclide(1, 3)
    cf252, pu239 = H.nuclide(98, 252), H.nuclide(94, 239)
    assert co60.radiological_score >= 9 > 8 > pu239.radiological_score >= 7
    assert cf252.sf_neutron_score > 0 and cf252.radiological_score > pu239.radiological_score
    assert tc99m.radiological_score < 4 and h3.radiological_score < 2
    assert H.nuclide(43, 99, -1).nuclide == "Tc-99"     # isotopia total -> ground state


def test_stable_and_product_score():
    mo98 = H.nuclide(42, 98)
    assert not mo98.radioactive and mo98.radiological_score == 0
    na24 = H.nuclide(11, 24)                            # strong gamma + reactive metal
    assert na24.product_score == pytest.approx(H.combine(na24.radiological_score, na24.chemical_score))
    assert na24.product_class == "7 – Severe"
    assert H.hazard_class(8.0) == "6 – Very high" and H.hazard_class(7.999) == "6 – Very high"


@pytest.mark.skipif(not LIBS.is_dir(), reason="isotopia.libs not found")
def test_products_of_a_run(tmp_path: Path):
    from isotopia_py.run import run
    inp = ISOTOPIA / "samples" / "g-Mo100-Mo099" / "org" / "isotopia.inp"
    st = run(inp.read_text(encoding="utf-8").splitlines(), out=io.StringIO(), outdir=str(tmp_path))
    prods = H.products(st)
    keys = [(h.Z, h.A, h.isomer) for h, _ in prods]
    assert (42, 99, 0) in keys and len(keys) == len(set(keys))
    assert all(a >= 0 for _, a in prods)
    assert [a for _, a in prods] == sorted((a for _, a in prods), reverse=True)
    w = H.activity_weighted(st)
    assert w is not None and 0 <= w <= 10
    assert H.for_state(st).Z == 42
