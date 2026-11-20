"""Tests for the 'layer' keyword: the numbered stack of capsule layers and the target.

These need no cross-section library: they run the input checks, the stopping-power set-up
(run.reacinitial -> rates.setup_stopping), the beam spectrum and the summary table.
"""

from __future__ import annotations

import importlib
import io

import pytest

from isotopia_py import output, rates, srim
from isotopia_py.constants import RHOTABLE
from isotopia_py.energyloss import RangeTable
from isotopia_py.inputdeck import checkvalue, read_keywords, readinput
from isotopia_py.run import reacinitial
from isotopia_py.state import IsotopiaError, State

run_module = importlib.import_module("isotopia_py.run")  # the module, not the function
MO100 = ("projectile p", "element Mo", "mass 100")
INDEX = srim.table_inputs()


def _setup(*lines):
    """Read and check the input, set the density, follow the beam through the stack."""
    st = State(out=io.StringIO())
    readinput(st, list(lines))
    read_keywords(st)
    checkvalue(st)
    reacinitial(st)
    return st


def _leff(st):
    """Effective target thickness [cm] from the beam spectrum (manual Eq. 4.23)."""
    rates._charged_particle_spectrum(st)
    return float(st.Leff)


def _element_table(particle, z, rho=None):
    """Independent RangeTable of a pure element at the data sheet's density and atomic weight."""
    rho = float(RHOTABLE[z - 1]) if rho is None else rho
    return RangeTable.from_srim(
        srim.load_table(particle, z), rho, float(INDEX[z]["target_mass_u"])
    )


# ---------------------------------------------------------------------------------------------
# Layers before the target: the beam reaches the target with less energy
# ---------------------------------------------------------------------------------------------


def test_front_layer_lowers_the_energy_at_the_target():
    st = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.0254", "layer 2 target", "thickness 0.02")
    expected = float(_element_table("p", 13).exit_energy(18.0, 0.0254))
    assert float(st.Ebeam_incident) == 18.0
    assert float(st.Ebeam) == pytest.approx(expected, rel=1e-6)
    (layer,) = st.layers
    assert (layer.number, layer.Z, layer.E_in) == (1, 13, 18.0)
    assert layer.E_out == pytest.approx(expected, rel=1e-12)
    assert layer.rho == float(RHOTABLE[13 - 1]) and layer.mass == float(
        INDEX[13]["target_mass_u"]
    )


def test_stack_equals_a_bare_target_hit_at_the_degraded_energy():
    stack = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.0254", "layer 2 target", "thickness 0.02")
    bare = _setup(*MO100, f"Ebeam {float(stack.Ebeam)!r}", "thickness 0.02")
    assert float(stack.Eback) == pytest.approx(float(bare.Eback), rel=1e-6)
    assert _leff(stack) == pytest.approx(_leff(bare), rel=1e-6)


def test_layer_thickness_and_energies_agree_both_ways():
    """Same rule as for the target: the thickness that takes 18 -> 15 MeV gives back 15 MeV."""
    thickness = float(_element_table("p", 13).thickness(18.0, 15.0))
    st = _setup(*MO100, "Ebeam 18.", f"layer 1 Al {thickness!r}", "layer 2 target", "thickness 0.02")
    assert float(st.Ebeam) == pytest.approx(15.0, rel=1e-6)


def test_two_layers_equal_one_layer_of_the_summed_thickness():
    one = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.03", "layer 2 target", "thickness 0.02")
    two = _setup(
        *MO100,
        "Ebeam 18.",
        "layer 1 Al 0.01",
        "layer 2 Al 0.02",
        "layer 3 target",
        "thickness 0.02",
    )
    assert float(two.Ebeam) == pytest.approx(float(one.Ebeam), rel=1e-6)
    assert [layer.E_in for layer in two.layers][1] == two.layers[0].E_out


def test_layers_are_crossed_in_number_order_whatever_the_input_order():
    st = _setup(
        *MO100,
        "Ebeam 30.",
        "layer 3 target",
        "layer 2 Cu 0.02",
        "layer 1 Al 0.05",
        "thickness 0.02",
    )
    after_al = float(_element_table("p", 13).exit_energy(30.0, 0.05))
    after_cu = float(_element_table("p", 29).exit_energy(after_al, 0.02))
    assert float(st.Ebeam) == pytest.approx(after_cu, rel=1e-6)


def test_layer_density_can_be_given():
    """Half the density and twice the thickness hold the same atoms per cm2."""
    full = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.02", "layer 2 target", "thickness 0.02")
    half = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.04 1.35", "layer 2 target", "thickness 0.02")
    assert half.layers[0].rho == 1.35
    assert float(half.Ebeam) == pytest.approx(float(full.Ebeam), rel=1e-6)


def test_gas_layer_and_alpha_beam():
    st = _setup(
        "projectile a",
        "element Au",
        "mass 197",
        "Ebeam 40.",
        "layer 1 He 10.",
        "layer 2 target",
        "Eback 30.",
    )
    expected = float(_element_table("a", 2).exit_energy(40.0, 10.0))
    assert 39.0 < expected < 40.0
    assert float(st.Ebeam) == pytest.approx(expected, rel=1e-6)


def test_default_eback_is_five_mev_below_the_energy_at_the_target():
    st = _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.0254", "layer 2 target")
    assert float(st.Eback) == pytest.approx(float(st.Ebeam) - 5.0, rel=1e-6)
    with pytest.raises(
        IsotopiaError, match="Eback must be below the energy at the target"
    ):
        _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.0254", "layer 2 target", "Eback 17.")


def test_without_layers_nothing_changes():
    st = _setup(*MO100, "Ebeam 16.", "Eback 12.")
    assert st.layers == [] and st.Ebeam_incident == st.Ebeam and not st.beam_stops
    alone = _setup(*MO100, "Ebeam 16.", "Eback 12.", "layer 1 target")
    assert alone.layers == [] and alone.Ebeam == st.Ebeam and alone.Eback == st.Eback


# ---------------------------------------------------------------------------------------------
# Layers after the target: what leaves the target
# ---------------------------------------------------------------------------------------------


def test_layers_after_the_target_start_from_eback():
    st = _setup(
        *MO100,
        "Ebeam 18.",
        "thickness 0.02",
        "layer 1 target",
        "layer 2 Al 0.0254",
        "layer 3 Cu 0.01",
    )
    al, cu = st.layers
    assert al.E_in == float(st.Eback)
    assert al.E_out == pytest.approx(
        float(_element_table("p", 13).exit_energy(al.E_in, 0.0254)), rel=1e-12
    )
    assert cu.E_in == al.E_out and 0.0 < cu.E_out < cu.E_in
    assert st.out.getvalue() == ""


def test_beam_stopping_in_a_layer_after_the_target():
    st = _setup(
        *MO100,
        "Ebeam 18.",
        "thickness 0.02",
        "layer 1 target",
        "layer 2 Cu 0.3",
        "layer 3 Al 0.1",
    )
    cu, al = st.layers
    reach = float(_element_table("p", 29).path_length(cu.E_in))
    assert (
        cu.E_out == 0.0 and cu.reach == pytest.approx(reach, rel=1e-12) and reach < 0.3
    )
    assert not al.reached
    assert "ISOTOPIA-warning: the beam stops in layer 2 (Cu)" in st.out.getvalue()


def test_beam_stopping_in_the_target_never_reaches_the_layers_behind():
    st = _setup(*MO100, "Ebeam 16.", "thickness 0.1", "layer 1 target", "layer 2 Al 0.0254")
    assert st.beam_stops and not st.layers[0].reached
    assert "the beam stops in the target" in st.out.getvalue()


def test_beam_stopping_before_the_target_is_an_error():
    with pytest.raises(
        IsotopiaError, match="stops in layer 1 .Al.*no beam reaches"
    ):
        _setup(*MO100, "Ebeam 18.", "layer 1 Al 0.5", "layer 2 target", "thickness 0.02")


# ---------------------------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------------------------


def test_summary_table_lists_the_stack_in_beam_order():
    st = _setup(
        *MO100,
        "Ebeam 18.",
        "Ibeam 0.15",
        "layer 3 Cu 0.3",
        "layer 1 Al 0.0254",
        "thickness 0.02",
        "layer 2 target",
        "layer 4 Al 0.1",
    )
    st.Ibeam = 1.5e-4  # A; set by inputdeck.conversion in a full run
    st.targetnuclide = "Mo100"
    _leff(st)
    st.out = io.StringIO()
    output._layers_out(st)
    rows = [
        ln.split() for ln in st.out.getvalue().splitlines() if ln[:6].strip().isdigit()
    ]
    assert [(r[0], r[1]) for r in rows] == [
        ("1", "Al"),
        ("2", "Mo100"),
        ("3", "Cu"),
        ("4", "Al"),
    ]
    assert rows[1][-1] == "target"
    # heat in layer 1 [kW] = 1000 * Ibeam [A] * energy lost [MeV] / charge number
    front = st.layers[1]
    assert float(rows[0][6]) == pytest.approx(
        0.15 * (front.E_in - front.E_out), rel=1e-5
    )
    text = st.out.getvalue()
    assert "beam stops at" in text and "not reached by the beam" in text


# ---------------------------------------------------------------------------------------------
# Isotope production in the layers (the runs themselves need the library: test_samples.py)
# ---------------------------------------------------------------------------------------------


def test_each_reached_layer_is_run_as_a_natural_target(monkeypatch, tmp_path):
    lines = [
        *MO100,
        "Ebeam 18.",
        "Ibeam 0.15",
        "thickness 0.02",
        "layer 1 Al 0.0254",
        "layer 2 target",
        "layer 3 Cu 0.3",
        "layer 4 Al 0.1",
    ]
    st = _setup(*lines)
    st.outdir, st.out = str(tmp_path), io.StringIO()
    decks = []

    def missing_library(deck, out=None, outdir="."):
        decks.append((deck, outdir))
        raise IsotopiaError(" ISOTOPIA-error: Cross section file does not exist: x")

    monkeypatch.setattr(run_module, "run", missing_library)
    run_module.layer_production(st, lines)
    al, cu, _ = sorted(st.layers, key=lambda layer: layer.number)
    # one run per layer the beam reaches, in its own folder, with the beam lines kept
    assert [outdir for _, outdir in decks] == [
        str(tmp_path / "layer1_Al"),
        str(tmp_path / "layer3_Cu"),
    ]
    assert decks[0][0] == [
        "projectile p",
        "Ibeam 0.15",
        "element Al",
        "mass 0",
        "Ebeam 18.0",
        f"Eback {al.E_out!r}",
        f"rho {al.rho!r}",
        "stopping srim",
    ]
    # the beam stops in the Cu layer, so that run is given the thickness
    assert f"Ebeam {cu.E_in!r}" in decks[1][0] and "thickness 0.3" in decks[1][0]
    # missing data are a warning in the main output, not an error
    text = st.out.getvalue()
    assert "ISOTOPIA-warning: no production calculated for layer 1 (Al):" in text
    assert "Cross section file does not exist: x" in text
    assert "Layer 4 (Al): not reached by the beam, no production" in text


# ---------------------------------------------------------------------------------------------
# Input errors
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines,message",
    [
        (
            ("projectile n", "element Mo", "mass 98", "layer 1 Al 0.1", "layer 2 target"),
            "charged-particle",
        ),
        ((*MO100, "Ebeam 18.", "layer 1 Al 0.1"), "say which layer is the target"),
        ((*MO100, "Ebeam 18.", "layer front Al 0.1"), "Error in layer front al"),
        (
            (*MO100, "Ebeam 18.", "layer 1 Al 0.1", "layer 3 target"),
            "numbered 1, 2, 3, ... along the beam, each number once; the input has 1, 3",
        ),
        (
            (*MO100, "Ebeam 18.", "layer 1 Al 0.1", "layer 1 Cu 0.1", "layer 2 target"),
            "each number once; the input has 1, 1, 2",
        ),
        (
            (*MO100, "Ebeam 18.", "layer 1 target", "layer 2 target"),
            "the target is given twice",
        ),
        ((*MO100, "Ebeam 18.", "layer 1 Xx 0.1", "layer 2 target"), "layer 1: wrong element"),
        ((*MO100, "Ebeam 18.", "layer 1 Al", "layer 2 target"), "Error in layer 1 al"),
        ((*MO100, "Ebeam 18.", "layer 1 Al -0.1", "layer 2 target"), "0 < thickness"),
        ((*MO100, "Ebeam 18.", "layer 1 Al 0.1 500.", "layer 2 target"), "0 < density"),
        (
            (*MO100, "Ebeam 18.", "layer 1 Np 0.01", "layer 2 target"),
            "layer 1: no SRIM table for p in Np",
        ),
        (
            (*MO100, "Ebeam 18.", "layer 1 Fm 0.01", "layer 2 target"),
            "no verified density for Fm",
        ),
        (
            (
                "projectile d",
                "element Mo",
                "mass 100",
                "Ebeam 18.",
                "layer 1 target",
                "layer 2 Np 0.01",
            ),
            "layer 2: no SRIM table for d in Np",
        ),
    ],
)
def test_layer_input_errors(lines, message):
    with pytest.raises(IsotopiaError, match=message):
        _setup(*lines)
