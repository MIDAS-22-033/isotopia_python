"""Run the ISOTOPIA sample cases through the Python port and compare them
with the reference outputs shipped in isotopia/samples/*/org.

Needs the cross-section library (isotopia.libs in this repository, or ISOTOPIA_LIBS)
and the samples folder next to files/ (in this repository, or ISOTOPIA_DIR). Without it the
sample tests are skipped.

    pytest                      # every sample (a few minutes)
    pytest -k Mo100             # one sample
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from isotopia_py.compare import compare_dirs, summarize
from isotopia_py.inputdeck import data_directories
from isotopia_py.run import run

ROOT = Path(__file__).resolve().parents[1]
# Data folders as the code finds them: environment variables, else inside this repository,
# else the Fortran's sibling folders. The samples are next to files/.
ISOTOPIA, LIBS = (Path(d) for d in data_directories())
SAMPLES = ISOTOPIA / "samples"

# Files where the Fortran itself differs between platforms: a stable nuclide
# fed only by decay of a very long-lived parent, computed as the difference of
# nearly equal terms. Compared more loosely.
LOOSE = {"Ru099.act": 1e-2}


def _cases():
    if not SAMPLES.is_dir():
        return []
    return sorted(d.name for d in SAMPLES.iterdir() if (d / "org" / "isotopia.inp").is_file())


def _libs_available() -> bool:
    return LIBS.is_dir()


def _run(case: str, outdir: Path):
    inp = SAMPLES / case / "org" / "isotopia.inp"
    # The reference outputs were made with the Fortran's density table and Bethe-Bloch
    # stopping power. The port's defaults are now the group's 25 degC densities and SRIM
    # stopping tables, so the samples are run with fortran_defaults.
    with open(outdir / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
        run(inp.read_text(encoding="utf-8").splitlines(), out=out, outdir=str(outdir),
            fortran_defaults=True)


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
@pytest.mark.parametrize("case", _cases())
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
@pytest.mark.parametrize("case", ["p-Mo100-Tc099m", "p-Ga000-Ge068"])
def test_thickness_input_reproduces_eback_run(case: str, tmp_path: Path):
    """Not in the Fortran: a target given as Ebeam and thickness must give the same Eback and
    the same results as the run with Ebeam and Eback that printed that thickness (SRIM
    stopping, the port's default; the second case is a natural target)."""
    inp = SAMPLES / case / "org" / "isotopia.inp"
    if not inp.is_file():
        pytest.skip(f"sample {case} not found")
    lines = inp.read_text(encoding="utf-8").splitlines()
    dirs = {name: tmp_path / name for name in ("eback", "thickness")}
    states = {}
    for name, d in dirs.items():
        d.mkdir()
        if name == "thickness":
            lines = [ln for ln in lines if not ln.lower().startswith("eback")]
            lines.append(f"thickness {float(states['eback'].Leff)!r}")
        with open(d / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
            states[name] = run(lines, out=out, outdir=str(d))
    assert float(states["thickness"].Eback) == pytest.approx(float(states["eback"].Eback), rel=1e-6)
    # isotopia.out echoes the input, which differs by design; every result file must agree
    results = compare_dirs(dirs["thickness"], dirs["eback"])
    bad = [f"{r.name}: {r.status} {r.worst}" for r in results
           if r.status != "ok" and r.name != "isotopia.out"]
    assert len(results) > 10 and not bad, "\n".join(bad)


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
@pytest.mark.parametrize("case", ["p-Mo100-Tc099m", "p-Ga000-Ge068"])
def test_target_behind_a_layer_equals_bare_target_at_the_degraded_energy(case: str, tmp_path: Path):
    """Not in the Fortran: a target behind a layer must give the same results as the bare
    target hit at the energy the beam has left after that layer. Layers behind the target
    must not change the target's results (the second case is a natural target)."""
    inp = SAMPLES / case / "org" / "isotopia.inp"
    if not inp.is_file():
        pytest.skip(f"sample {case} not found")
    lines = inp.read_text(encoding="utf-8").splitlines()
    rest = [ln for ln in lines if not ln.lower().startswith("ebeam")]   # keeps the sample's Eback
    ebeam = float(next(ln for ln in lines if ln.lower().startswith("ebeam")).split()[1])
    dirs = {name: tmp_path / name for name in ("stack", "bare")}
    states = {}
    for name, d in dirs.items():
        d.mkdir()
        if name == "stack":
            deck = rest + [f"Ebeam {ebeam + 2.0}", "layer 1 Al 0.0254", "layer 2 target",
                           "layer 3 Al 0.0254"]
        else:
            deck = rest + [f"Ebeam {float(states['stack'].Ebeam)!r}"]
        with open(d / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
            states[name] = run(deck, out=out, outdir=str(d))
    assert float(states["stack"].Ebeam) < ebeam + 2.0
    assert float(states["stack"].Eback) == float(states["bare"].Eback)
    # isotopia.out echoes the input and the layer table; every result file must agree
    results = compare_dirs(dirs["stack"], dirs["bare"])
    bad = [f"{r.name}: {r.status} {r.worst}" for r in results
           if r.status != "ok" and r.name != "isotopia.out"]
    assert len(results) > 10 and not bad, "\n".join(bad)


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
def test_natural_abundance_weighting(tmp_path: Path):
    """Natural Ga (60.1% Ga-69, 39.9% Ga-71): the merged inventory of every
    nuclide must be the abundance-weighted sum of the per-isotope runs. The
    Fortran violated this, before its equations.f90 fix, for nuclides made by
    only one isotope, where the other isotope's run re-used the stale
    inventory."""
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


def _mo100(tmp_path: Path, name: str, *lines: str, **options):
    """Run p + Mo-100 with extra input lines; returns (State, output directory)."""
    d = tmp_path / name
    d.mkdir()
    deck = ["projectile p", "element Mo", "mass 100", *lines]
    with open(d / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
        return run(deck, out=out, outdir=str(d), **options), d


def _atoms_per_second(st) -> dict:
    """Production rate of each product in the whole target [atoms/s]."""
    return {key: float(rate) * float(st.N_0) for key, rate in st.reaction_rate.items()
            if st.rpexist[key]}


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
def test_thin_target_between_tabulated_energies_keeps_its_products(tmp_path: Path):
    """Not in the Fortran: the window 15.9 -> 15.2 MeV holds no tabulated energy of the
    p + Mo-100 residual files (they step by 1 MeV here), so the Fortran rule drops the
    products. Splitting a target in two must not change what is produced: the rates of the
    windows 16.2 -> 15.9 and 15.9 -> 15.2 add up to those of 16.2 -> 15.2."""
    if not (LIBS / "p" / "Mo100").is_dir():
        pytest.skip("p + Mo100 library not found")
    upper = _atoms_per_second(_mo100(tmp_path, "upper", "Ebeam 16.2", "Eback 15.9")[0])
    thin = _atoms_per_second(_mo100(tmp_path, "thin", "Ebeam 15.9", "Eback 15.2")[0])
    both = _atoms_per_second(_mo100(tmp_path, "both", "Ebeam 16.2", "Eback 15.2")[0])
    assert sorted(thin) == sorted(both) and len(both) > 20
    for key, rate in both.items():
        assert upper[key] + thin[key] == pytest.approx(rate, rel=1e-4), key
    # the Fortran rule is kept for the comparison with the Fortran
    old = _mo100(tmp_path, "old", "Ebeam 15.9", "Eback 15.2", fortran_defaults=True)[0]
    assert len(_atoms_per_second(old)) < len(both)


@pytest.mark.skipif(not _libs_available(), reason="isotopia.libs not found")
def test_production_in_the_layers(tmp_path: Path):
    """Not in the Fortran: each layer the beam reaches is run as a natural-element target at
    the energies of the stack table; the folder of a layer holds the same results as running
    that layer on its own."""
    if not all((LIBS / "p" / n).is_dir() for n in ("Mo100", "Al027", "Cu063", "Cu065")):
        pytest.skip("p + Mo100, Al027, Cu063, Cu065 libraries not found")
    beam = ("Ibeam 0.15", "Tirrad 6 h")
    st, d = _mo100(tmp_path, "stack", "Ebeam 18.", "thickness 0.02", *beam, "layer 1 Al 0.0254",
                   "layer 2 target", "layer 3 Cu 0.3", "layer 4 Al 0.1")
    al, cu, unreached = sorted(st.layers, key=lambda layer: layer.number)
    # natural Al has one isotope, so its files keep the isotope suffix (as in the Fortran)
    assert (d / "layer1_Al" / "Si027.act.027").is_file()
    assert (d / "layer3_Cu" / "Zn065.act").is_file()
    assert not (d / "layer4_Al").exists() and not unreached.reached
    text = (d / "isotopia.out").read_text(encoding="utf-8")
    summary = text[text.index(" Isotope production in the layers"):]
    assert "Layer 1 (Al)" in summary and "Zn  65" in summary
    assert "Layer 4 (Al): not reached by the beam" in summary
    # the layers leave the target's own results as they are without the summary
    bare, dbare = _mo100(tmp_path, "bare", f"Ebeam {float(st.Ebeam)!r}", "thickness 0.02", *beam)
    bad = [r.name for r in compare_dirs(d, dbare) if r.status != "ok" and r.name != "isotopia.out"]
    assert not bad, bad
    # layer 1 on its own: natural Al between the energies of the stack table
    alone = tmp_path / "alone"
    alone.mkdir()
    with open(alone / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
        run(["projectile p", "element Al", "mass 0", f"Ebeam {al.E_in!r}", f"Eback {al.E_out!r}",
             *beam], out=out, outdir=str(alone))
    bad = [r.name for r in compare_dirs(d / "layer1_Al", alone)
           if r.status != "ok" and r.name != "isotopia.out"]
    assert not bad, bad
    # the beam stops in the Cu layer: its run covers the energy at its face down to the floor
    assert cu.E_out == 0.0 and "the beam stops in the target" in (
        d / "layer3_Cu" / "isotopia.out").read_text(encoding="utf-8")
