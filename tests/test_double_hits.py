"""Double hitting in the main run (on by default; keyword 'hits').

1. Made-up data: nuclides without cross sections are split by half-life. Unstable
   and shorter-lived than a day: assumed not hit. Stable, a day or longer, or no
   decay data: reported as missing cross sections.
2. Keyword and defaults: 'hits' is on by default, 'hits n' turns it off, and
   fortran_defaults turns it off so the Fortran comparisons are unchanged.
3. p + Mo-100 sample (needs isotopia/ and isotopia.libs/, else skipped): with hits,
   the run's own inventories carry products made only by second hits (Tc-98) and
   equal daughters.irradiate_with_hits; without hits they equal the port; the main
   output and missing_cross_sections.txt report the double hitting.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
from fakes import LIBS, SAMPLES, FakeLibrary

from isotopia_py.daughters import SHORT_LIVED, classify_no_data, irradiate_with_hits, port_window
from isotopia_py.output import MISSING_XS_FILE
from isotopia_py.run import run

# ----------------------------------------------------------------------------
# 1. Made-up data
# ----------------------------------------------------------------------------

def test_classify_no_data_by_half_life():
    hour, day = 3600.0, 86400.0
    short, long_, stable, unknown, edge = (30, 64, 0), (30, 65, 0), (30, 66, 0), (30, 99, 0), (30, 70, 0)
    library = FakeLibrary({short: (math.log(2) / (12.7 * hour), []),
                           long_: (math.log(2) / (244 * day), []),
                           edge: (math.log(2) / SHORT_LIVED, []),
                           stable: (0.0, [])})
    short_lived, missing = classify_no_data([unknown, stable, long_, short, edge], library)
    assert [n for n, _ in short_lived] == [short]
    assert short_lived[0][1] == pytest.approx(12.7 * hour)
    # a day exactly is not short-lived; stable and unknown half-lives are reported
    assert [n for n, _ in missing] == [long_, stable, edge, unknown]
    assert dict(missing)[stable] == math.inf and dict(missing)[unknown] is None


# ----------------------------------------------------------------------------
# 2. Keyword and defaults (no library needed: only the input deck is read)
# ----------------------------------------------------------------------------

def _flag(lines, **kw):
    from isotopia_py import inputdeck
    from isotopia_py.state import State
    st = State(out=io.StringIO(), **kw)
    inputdeck.readinput(st, lines)
    inputdeck.read_keywords(st)
    return st.flaghits


DECK = ["projectile p", "element Mo", "mass 100", "Ebeam 16.", "Eback 12."]


def test_hits_keyword_and_defaults():
    assert _flag(DECK) is True
    assert _flag(DECK + ["hits n"]) is False
    assert _flag(DECK + ["hits y"]) is True
    assert _flag(DECK, fortran_defaults=True) is False
    assert _flag(DECK + ["hits y"], fortran_defaults=True) is True


# ----------------------------------------------------------------------------
# 3. p + Mo-100
# ----------------------------------------------------------------------------

MO100 = SAMPLES / "p-Mo100-Tc099m" / "org" / "isotopia.inp"
needs_mo100 = pytest.mark.skipif(not (MO100.is_file() and (LIBS / "p" / "Mo100").is_dir()
                                      and (LIBS / "p" / "Tc099").is_dir()),
                                 reason="needs the p-Mo100-Tc099m sample and the p + Mo100, Tc099 libraries")
TC98, TC99M, MO99 = (43, 98, -1), (43, 99, 1), (42, 99, -1)


@needs_mo100
def test_mo100_run_uses_double_hitting(tmp_path):
    lines = MO100.read_text(encoding="utf-8").splitlines()
    dirs = {name: tmp_path / name for name in ("hits", "single")}
    st = {}
    for name, extra in (("hits", []), ("single", ["hits n"])):
        dirs[name].mkdir()
        out = io.StringIO()
        st[name] = run(lines + extra, out=out, outdir=str(dirs[name]))
        st[name].text = out.getvalue()
    hits, single = st["hits"], st["single"]
    assert hits.flaghits and not single.flaghits

    # Tc-98 is made only by second hits (Tc-99(p,pn), Mo-99(p,2n), ...)
    assert float(single.Niso[TC98][single.Ntime]) == 0.0
    assert float(hits.Niso[TC98][hits.Ntime]) > 0.0 and hits.Yexist[TC98]
    # Main products barely change: second-hit corrections are of order R t_irr
    for key in (TC99M, MO99):
        assert float(hits.Niso[key][hits.Ntime]) == pytest.approx(
            float(single.Niso[key][single.Ntime]), rel=5e-3)

    # The run's inventories are those of irradiate_with_hits on the same grid
    inv = irradiate_with_hits(hits, port_window(hits), times=[float(t) for t in hits.Tgrid],
                              on_missing="warn")
    tc98 = sum(inv.atoms[:, k] for k, n in enumerate(inv.nuclides) if n[:2] == (43, 98))
    np.testing.assert_allclose(np.asarray(hits.Niso[TC98], dtype=float), tc98, rtol=1e-6)

    # Reported with the outputs
    assert "Double hitting (not in the Fortran)" in hits.text
    assert "Double hitting" not in single.text
    assert hits.hits_report and hits.hits_report["hit"] > 1
    report = (dirs["hits"] / MISSING_XS_FILE).read_text(encoding="utf-8")
    assert report.startswith("# Nuclides in the double-hitting window")
    assert len(report.splitlines()) == 3 + len(hits.hits_report["missing"])
    assert not (dirs["single"] / MISSING_XS_FILE).exists()
