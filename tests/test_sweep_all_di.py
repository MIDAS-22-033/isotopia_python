"""Tests for scripts/sweep_all_di.py: the all-DI sweep must agree with the single-DI
search (optimal_route_v4_step_beam.run_one) and with ISOTOPIA's own State."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sweep_all_di", ROOT / "scripts" / "sweep_all_di.py")
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep_all_di"] = sweep          # dataclasses need the module registered
_spec.loader.exec_module(sweep)
opt = sweep.opt

from isotopia_py import run_cache

LIBS = Path(os.environ.get("ISOTOPIA_LIBS", ROOT.parent / "isotopia.libs"))
needs_lib = pytest.mark.skipif(not (LIBS / "p" / "Mo100").is_dir() or not os.environ.get("ISOTOPIA_DIR"),
                               reason="needs ISOTOPIA_DIR and the p + Mo100 library")
LINES = ["projectile p", "element Mo", "mass 100", "Ibeam 1", "Tirrad 1 d", "Tcool 1 d",
         "thickness 0.05", "Ebeam 20.000000"]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _settings():
    st = sweep.State()
    sweep.machine(st)
    return sweep.Settings(st.path, 0.05, 1.0, "1 d", "1 d")


def _spec_mo100(cache_dir=None, di=None):
    return opt.RunSpec(di, 2, opt.Target(42, 100, True), 20.0, None, list(LINES),
                       cache_dir=None if cache_dir is None else str(cache_dir))


@needs_lib
def test_sweep_row_matches_single_di_search():
    """For Tc-99m and Mo-99 the sweep's masses equal run_one's exactly, and the
    activities are ISOTOPIA's lam N at EOI and end of cooling."""
    s = _settings()
    info, rows, _ = sweep.sweep_one(_spec_mo100(), s)
    assert info["status"] == "ok" and info["n_products"] == len(rows) > 10
    by_name = {r["product"]: r for r in rows}
    st = run_cache.cached_run(LINES, None)
    for di, key in (((43, 99, 1), (43, 99, 1)), ((42, 99, 0), (42, 99, -1))):
        name = opt.nuclide_name(*di)
        single = opt.run_one(opt.RunSpec(di, 2, opt.Target(42, 100, True), 20.0, None, list(LINES),
                                         molar_mass=opt.molar_mass(s.path, di[0], di[1])))
        r = by_name[name]
        assert (r["m_rate"], r["m_EOI_g"], r["m_EOC_g"]) == (single["m_rate"], single["m_EOI_g"],
                                                             single["m_EOC_g"])
        assert r["Eback_MeV"] == single["Eback_MeV"] and r["stopping"] == single["stopping"] == "srim"
        lam = float(st.lam[key])
        assert r["half_life_s"] == pytest.approx(math.log(2) / lam)
        assert r["A_EOI_Bq"] == pytest.approx(lam * float(st.Niso[key][st.Ntime]), rel=1e-6)
        assert r["A_EOC_Bq"] == float(st.activity[key][opt.eoc_index(st)])


@needs_lib
def test_products_exclude_target_and_totals():
    """No row for the unreacted target, and a nuclide with isomers is reported per state,
    never also as its total, so masses are not counted twice."""
    st = run_cache.cached_run(LINES, None)
    found = list(sweep.products(st, opt.Target(42, 100, True)))
    assert all(I in (0, 1, 2) for _, I in found)
    assert (42, 100) not in {(k[0], k[1]) for k, I in found if I == 0}
    za = [(k[0], k[1]) for k, _ in found]
    for (Z, A) in set(za):
        n_iso = st.Nisomer[(Z, A)]
        assert za.count((Z, A)) <= (1 if n_iso == -1 else n_iso + 1)
    assert ((43, 99, 1), 1) in found and ((43, 99, 0), 0) in found


@needs_lib
def test_sweep_outputs_and_cache(tmp_path):
    """End to end on p + Mo100 at 10, 20, 30 MeV: the best-per-DI table holds the
    maximum of the full table, and a rerun reads every run from the cache with
    identical tables."""
    args = ["-p", "p", "--targets", "Mo100", "--emin", "10", "--emax", "30", "--de", "10",
            "--cache", str(tmp_path / "cache")]
    assert sweep.main(args + ["--out", str(tmp_path / "a")]) == 0
    with gzip.open(tmp_path / "a" / "full_table.csv.gz", "rt") as fh:
        full = list(csv.DictReader(fh))
    assert {r["Ebeam_MeV"] for r in full} == {"10.000000", "20.000000", "30.000000"}
    best = {r["product"]: r for r in _read_csv(tmp_path / "a" / "best_per_di.csv")}
    tc = [float(r["m_EOC_g"]) for r in full if r["product"] == "Tc99_m1"]
    assert float(best["Tc99_m1"]["m_EOC_g"]) == pytest.approx(max(tc), rel=1e-6)
    assert sweep.main(args + ["--out", str(tmp_path / "b")]) == 0
    assert "(3 from the cache, 0 computed" in (tmp_path / "b" / "summary.txt").read_text()
    for f in ("best_per_di.csv", "best_per_route.csv"):
        assert (tmp_path / "a" / f).read_text() == (tmp_path / "b" / f).read_text()
    runs = [_read_csv(tmp_path / d / "runs.csv") for d in "ab"]
    assert [r["from_cache"] for r in runs[1]] == ["True"] * 3
    for r in runs[0] + runs[1]:
        del r["from_cache"]
    assert runs[0] == runs[1]


@needs_lib
def test_estimate_reports_counts(tmp_path):
    s = _settings()
    specs = [opt.RunSpec(None, 2, opt.Target(42, 100, True), e, None,
                         LINES[:-1] + [f"Ebeam {e:.6f}"], cache_dir=str(tmp_path)) for e in (15.0, 25.0)]
    text = sweep.estimate(specs, s, 1, 0, 4, True)
    assert "Runs: 2 in total, 2 still to compute" in text and "with --jobs 4" in text
