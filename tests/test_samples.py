"""Run the ISOTOPIA sample cases through the Python port and compare them
with the reference outputs shipped in isotopia/samples/*/org.

Needs the cross-section library: set ISOTOPIA_DIR (and ISOTOPIA_LIBS when
isotopia.libs is not a sibling of the isotopia directory). Without it the
sample tests are skipped.

    pytest                      # every sample (a few minutes)
    pytest -k Mo100             # one sample
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from isotopia_py.compare import compare_dirs, summarize
from isotopia_py.run import run

ROOT = Path(__file__).resolve().parents[1]
ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
SAMPLES = ISOTOPIA / "samples"

# Files where the Fortran itself differs between platforms: a stable nuclide
# fed only by decay of a very long-lived parent, computed as the difference of
# nearly equal terms. Compared more loosely.
LOOSE = {"Ru099.act": 1e-2}

# Natural-element targets (mass 0). The port clears nuclide inventories
# between the per-isotope runs, which the Fortran does not (see README), so
# these cases differ from the Fortran references on purpose. They are checked
# by test_natural_abundance_weighting instead.
NATURAL = {"p-Ga000-Ge068", "p-Rb000-Sr082"}


def _cases():
    if not SAMPLES.is_dir():
        return []
    return sorted(d.name for d in SAMPLES.iterdir() if (d / "org" / "isotopia.inp").is_file())


def _libs_available() -> bool:
    libs = os.environ.get("ISOTOPIA_LIBS") or str(ISOTOPIA.parent / "isotopia.libs")
    return Path(libs).is_dir()


def _run(case: str, outdir: Path):
    inp = SAMPLES / case / "org" / "isotopia.inp"
    with open(outdir / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
        run(inp.read_text(encoding="utf-8").splitlines(), out=out, outdir=str(outdir))


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
@pytest.mark.parametrize("case", [c for c in _cases() if c not in NATURAL])
def test_sample(case: str, tmp_path: Path):
    _run(case, tmp_path)
    results = compare_dirs(tmp_path, SAMPLES / case / "org")
    bad = []
    for r in results:
        if r.status == "ok":
            continue
        if r.status == "numeric" and r.max_rel <= LOOSE.get(r.name, 0.0):
            continue
        if r.name == "isotopia.out" and r.status == "numeric" and r.max_rel <= 1e-3:
            continue  # the summary line of a LOOSE nuclide
        bad.append(f"{r.name}: {r.status} {r.worst} {r.notes[:2]}")
    assert not bad, summarize(results) + "\n" + "\n".join(bad)


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
def test_natural_abundance_weighting(tmp_path: Path):
    """Natural Ga (60.1% Ga-69, 39.9% Ga-71): the merged inventory of every
    nuclide must be the abundance-weighted sum of the per-isotope runs. The
    Fortran violates this for nuclides made by only one isotope, where the
    other isotope's run re-used the stale inventory."""
    from isotopia_py.io import read_act

    _run("p-Ga000-Ge068", tmp_path)
    abun = {"069": 0.60108, "071": 0.39892}
    checked = 0
    for nat in tmp_path.glob("*.act"):
        total = read_act(nat).column("isotope_inventory")
        expected = [0.0] * len(total)
        for ext, ab in abun.items():
            per = tmp_path / (nat.name + "." + ext)
            if per.exists():
                for k, v in enumerate(read_act(per).column("isotope_inventory")):
                    expected[k] += ab * v
        for t, e in zip(total, expected):
            assert t == pytest.approx(e, rel=2e-6, abs=1e-30), nat.name
        checked += 1
    assert checked > 50
