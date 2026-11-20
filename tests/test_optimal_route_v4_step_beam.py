"""Tests for scripts/optimal_route_v4_step_beam.py.

The first group needs no data. The last group needs ISOTOPIA_DIR and the proton
library for Mo-100 and Ru-101 (skipped otherwise).
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "optimal_route", ROOT / "scripts" / "optimal_route_v4_step_beam.py")
opt = importlib.util.module_from_spec(_spec)
sys.modules["optimal_route"] = opt          # dataclasses need the module registered
_spec.loader.exec_module(opt)


# ---------------------------------------------------------------- names and lists

def test_parse_and_display_names():
    assert opt.parse_nuclide("Tc99_m1") == (43, 99, 1)
    assert opt.parse_nuclide("Tc99_g") == (43, 99, 0)
    assert opt.parse_nuclide("Tc99") == (43, 99, None)
    assert opt.nuclide_name(43, 99, 1) == "Tc99_m1"
    assert opt.nuclide_name(43, 99, None) == "Tc99"
    for bad in ("tc99", "Tc099m", "Xx99", "Tc99_m3"):
        with pytest.raises(ValueError):
            opt.parse_nuclide(bad)


def test_read_nuclide_list(tmp_path):
    f = tmp_path / "exclude.txt"
    f.write_text("# blacklist\nMo99   # comment\nTc99_m1, Ru99_g\n\n")
    assert opt.read_nuclide_list(f) == {(42, 99, None), (43, 99, 1), (44, 99, 0)}
    f.write_text("Mo99\nnot_a_nuclide\n")
    with pytest.raises(ValueError, match="line 2"):
        opt.read_nuclide_list(f)


def test_listed_matching_rules():
    entries = {(42, 99, None), (43, 99, 1)}
    assert opt.listed(42, 99, 0, entries)            # bare entry matches every state
    assert opt.listed(42, 99, 1, entries)
    assert opt.listed(43, 99, 1, entries)
    assert not opt.listed(43, 99, 0, entries)        # Tc99_m1 does not match the ground state
    assert opt.listed(43, 99, -1, entries)           # ISOTOPIA 'total' matches any listed state
    assert not opt.listed(44, 99, -1, entries)


def test_energy_grid():
    g = opt.energy_grid(0.001, 200.0, 1.0)
    assert g[0] == pytest.approx(0.001) and g[-1] == pytest.approx(199.001) and len(g) == 200
    assert opt.energy_grid(1.0, 3.0, 0.5).tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_candidate_targets_mass_window(tmp_path):
    """Targets span A-5..A+5; lighter ones that cannot reach A_DI (A + 1 < A_DI
    for protons) are skipped with a warning, isomeric targets are skipped."""
    lib = tmp_path / "libs"
    for name in ("Mo094", "Mo098", "Mo100", "Ru104", "Ru105", "Tc099m"):
        (lib / "p" / name).mkdir(parents=True)
    ab = tmp_path / "code" / "files" / "abundance"
    ab.mkdir(parents=True)
    (ab / "z042").write_text("".join(f"{42:4d}{a:4d}{10.0:11.6f}\n" for a in (94, 98, 100)))
    (ab / "z044").write_text(f"{44:4d}{104:4d}{10.0:11.6f}\n")
    warnings: list[str] = []
    targets = opt.candidate_targets((43, 99, 1), 2, str(tmp_path / "code"), str(lib), False, warnings)
    assert [t.name for t in targets] == ["Mo98", "Mo100", "Ru104"]
    assert any("Mo094" in w and "cannot reach" in w for w in warnings)
    assert any("Tc099m" in w for w in warnings)
    # The DI's own ground state is not a production route; its isomer from it is
    warnings.clear()
    (lib / "p" / "Tc099").mkdir()
    (ab / "z043").write_text(f"{43:4d}{99:4d}{10.0:11.6f}\n")
    names = [t.name for t in opt.candidate_targets((43, 99, 0), 2, str(tmp_path / "code"), str(lib),
                                                   False, warnings)]
    assert "Tc99" not in names and any("is the DI itself" in w for w in warnings)
    names = [t.name for t in opt.candidate_targets((43, 99, 1), 2, str(tmp_path / "code"), str(lib),
                                                   False, [])]
    assert "Tc99" in names


# ---------------------------------------------------------------- ranking metrics

def _row(target, ebeam, rate, eoi, eoc):
    return {"projectile": "p", "target": target, "stable": True, "Ebeam_MeV": ebeam,
            "Eback_MeV": ebeam - 5, "Leff_cm": 0.05, "m_rate": rate, "m_rate_unit": "g/(mA h)",
            "m_EOI_g": eoi, "m_EOC_g": eoc, "status": "ok", "note": ""}


def test_metric_choice_changes_ranking(tmp_path):
    """Route A wins on initial rate and EOI mass, route B on end-of-cooling mass;
    the summary names the winner of the chosen metric and lists all three."""
    rows = [_row("Aa10", 20.0, 9e-6, 1e-4, 1e-6), _row("Aa10", 30.0, 1e-6, 1e-5, 1e-7),
            _row("Bb10", 25.0, 1e-6, 5e-5, 3e-5)]
    for metric, winner in (("rate", "Aa10"), ("eoi", "Aa10"), ("eoc", "Bb10")):
        args = opt.parse_args(["Tc99_m1", "-p", "p", "--de", "1", "--metric", metric])
        text = opt.write_outputs(tmp_path / metric, "Tc99_m1", rows, [], args)
        optimal = next(ln for ln in text.splitlines() if ln.startswith("OPTIMAL"))
        assert f"OPTIMAL ({metric}): {winner}" in optimal
        assert "rate: Aa10(p,x)Tc99_m1 at Ebeam = 20 MeV" in text
        assert "eoc: Bb10(p,x)Tc99_m1 at Ebeam = 25 MeV" in text
    header = (tmp_path / "eoc" / "scan.csv").read_text().splitlines()[0]
    assert "m_EOC_g" in header and "m_rate" in header and "Bq" not in header


def test_molar_mass_from_endf_record(tmp_path):
    """AWR is read from the MF1/MT451 first line of the decay file; a nuclide
    without a record stops with LookupError instead of a guessed mass."""
    d = tmp_path / "files" / "decay"
    d.mkdir(parents=True)
    head = " 4.210000+4 9.900000+1         -1          0          2          11000 1451    1\n"
    (d / "z042").write_text(head)
    assert opt.molar_mass(str(tmp_path), 42, 100) == pytest.approx(99.0 * opt.NEUTRON_MASS_U)
    with pytest.raises(LookupError, match="Mo101"):
        opt.molar_mass(str(tmp_path), 42, 101)


# ---------------------------------------------------------------- rarity index

def _workbook(path):
    """Minimal copy of the team workbook layout with made-up prices (USD/g):
    min 0.01 (Mo), max 1e6 (Tc-99m), so the log index is ln(p/0.01)/ln(1e8)."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Price per gram"
    ws.append(["Z", "Symbol", "Element", "Row type", "Form / isotope priced",
               "Average market price (USD per gram)", "Market / region of price"])
    ws.append([42, "Mo", "Molybdenum", "Element (natural / commercial form)", "Mo", 0.01, "United States"])
    ws.append([43, "Tc", "Technetium", "Element (natural / commercial form)", "Tc", 100.0, "United States"])
    ws.append([43, "Tc", "Technetium", "Isotope variant (Tc-99m: medical imaging)", "Technetium-99m",
               1e6, "United States"])
    ws.append([44, "Ru", "Ruthenium", "Element (natural / commercial form)", "Ru", 1.0,
               "Europe (Rotterdam / European price)"])
    ws.append([45, "Rh", "Rhodium", "Element (natural / commercial form)", "Rh", None, "—"])
    ob = wb.create_sheet("US supply ranking")
    ob.append(["title"]); ob.append(["text"])
    ob.append(["W", None, None, None, 0.5]); ob.append(["assumed", None, None, None, 0.5]); ob.append([])
    ob.append(["Rank", "Z", "Symbol", "Element", "Score", "Category", "published", "Import reliance used (%)",
               "China share of imports (%)", "Russia share of imports (%)"])
    ob.append([None, 42, "Mo", "Molybdenum", None, None, None, 40, 10, 10])    # 0.4*(0.5+0.5*0.2) = 0.24
    ob.append([None, 44, "Ru", "Ruthenium", None, None, None, 100, None, None])  # 1.0*(0.5+0.5*0.5) = 0.75
    wb.save(path)
    return path


def test_rarity_index_formulas_and_fallbacks(tmp_path):
    """Values follow the sheet's formulas; each fallback in the team diagram raises a flag."""
    idx = opt.RarityIndex(_workbook(tmp_path / "w.xlsx"))
    span = np.log(1e8)
    v, f = idx.lookup(43, 99, 1, "price")                 # isotope row exists: no flag
    assert v == pytest.approx(1.0) and f == []
    v, f = idx.lookup(43, 99, 0, "price")                 # ground state: element row, flagged
    assert v == pytest.approx(np.log(100 / 0.01) / span) and "element-level" in f[0]
    v, f = idx.lookup(42, 100, 0, "obtainability")
    assert v == pytest.approx(0.4 * (0.5 + 0.5 * 0.2)) and f == []
    v, _ = idx.lookup(44, 101, 0, "obtainability")        # no China/Russia data: assumed share
    assert v == pytest.approx(0.75)
    _, f = idx.lookup(44, 101, 0, "price")                # non-U.S. market is flagged
    assert any("Europe" in x for x in f)
    v, f = idx.lookup(43, 99, 1, "obtainability")         # basis missing: other basis, flagged
    assert v == pytest.approx(1.0) and any("used price" in x for x in f)
    v, f = idx.lookup(45, 103, 0, "price")                # neither basis: 0, flagged
    assert v == 0.0 and any("assumed 0" in x for x in f)


def test_rarity_weight_reorders_routes(tmp_path):
    """Rarity scales each route's mass by (1 + dR)/2: a rarer target drops in the
    ranking but is kept and flagged; without --rarity the ranking is unchanged."""
    idx = opt.RarityIndex(_workbook(tmp_path / "w.xlsx"))
    rows = [_row("Ru101", 20.0, 1e-6, 2e-5, 2e-5), _row("Mo100", 20.0, 1e-6, 1e-5, 1e-5)]
    warnings: list[str] = []
    opt.apply_rarity(rows, (43, 99, 0), idx, "price", "m_EOC_g", warnings)
    r_di = np.log(100 / 0.01) / np.log(1e8)
    ru, mo = rows
    assert ru["delta_R"] == pytest.approx(r_di - np.log(100) / np.log(1e8))
    assert mo["rarity_weight"] == pytest.approx((1 + r_di) / 2)         # Mo is the cheapest: R = 0
    assert ru["score"] == pytest.approx(2e-5 * ru["rarity_weight"])
    args = opt.parse_args(["Tc99", "-p", "p", "--de", "1", "--rarity", "w.xlsx"])
    text = opt.write_outputs(tmp_path / "on", "Tc99_g", rows, warnings, args)
    assert next(ln for ln in text.splitlines() if ln.startswith("OPTIMAL")).count("Ru101") == 1
    # A DI cheaper than its target: negative delta, kept, flagged
    rows = [_row("Ru101", 20.0, 1e-6, 2e-5, 2e-5)]
    warnings.clear()
    opt.apply_rarity(rows, (42, 99, 0), idx, "price", "m_EOC_g", warnings)
    assert rows[0]["delta_R"] < 0 and rows[0]["status"] == "ok"
    assert any("rarer than the DI" in w for w in warnings)
    text = opt.write_outputs(tmp_path / "neg", "Mo99_g", rows, warnings, args)
    assert "target rarer than DI" in text


# ---------------------------------------------------------------- step beam: solver checks

def _problem(P, D, energies=(10.0, 20.0), P_T=None, N_T0=1e20, t_irr=1e5, t_cool=0.0, di=1):
    P = np.asarray(P, dtype=float)
    return opt.StepProblem(np.asarray(energies, float), P,
                           np.zeros(len(P)) if P_T is None else np.asarray(P_T, float),
                           np.asarray(D, dtype=float), N_T0, t_irr, t_cool, di)


def test_step_solver_matches_bateman_single_product():
    """One product, constant source P, decay lambda: N(T) = P/lambda (1 - e^-lambda T),
    then e^-lambda tc in cooling. With two steps (P1 until t1, then P2):
    N(T) = P1/lambda (e^-lambda (T-t1) - e^-lambda T) + P2/lambda (1 - e^-lambda (T-t1))."""
    lam, T, tc = 2e-5, 1e5, 3e4
    P1, P2 = 1e10, 4e10
    prob = _problem([[0, P1], [0, P2]], [[0, 0], [0, -lam]], t_irr=T, t_cool=tc)
    eoi, eoc = prob.evolve([10.0], [T])
    assert eoi[1] == pytest.approx(P1 / lam * (1 - np.exp(-lam * T)), rel=1e-10)
    assert eoc[1] == pytest.approx(eoi[1] * np.exp(-lam * tc), rel=1e-10)
    t1 = 0.3 * T
    eoi, _ = prob.evolve([10.0, 20.0], [t1, T - t1])
    exact = (P1 / lam * (np.exp(-lam * (T - t1)) - np.exp(-lam * T))
             + P2 / lam * (1 - np.exp(-lam * (T - t1))))
    assert eoi[1] == pytest.approx(exact, rel=1e-10)


def test_step_solver_stiff_chain_and_burnup():
    """Source into B (lambda_B = 1e3 /s, very short) -> C (lambda_C = 1e-5 /s): exact
    N_C = P/l_C (1 - e^-l_C t) - P (e^-l_B t - e^-l_C t)/(l_C - l_B); lambdas 8 decades
    apart test the matrix exponential on a stiff system. Target burnup: N_T = N_0 e^-R t."""
    lB, lC, T, P = 1e3, 1e-5, 1e5, 1e9
    D = [[0, 0, 0], [0, -lB, 0], [0, lB, -lC]]
    prob = _problem([[0, P, 0], [0, P, 0]], D, t_irr=T, di=2)
    N = prob.evolve([10.0], [T])[0]
    exact = P / lC * (1 - np.exp(-lC * T)) - P * (np.exp(-lB * T) - np.exp(-lC * T)) / (lC - lB)
    assert N[2] == pytest.approx(exact, rel=1e-9)
    N_T0, P_T = 1e20, 1e15
    prob = _problem([[0, 0], [0, 0]], [[0, 0], [0, 0]], P_T=[P_T, P_T], N_T0=N_T0, t_irr=T)
    assert prob.evolve([10.0], [T])[0][0] == pytest.approx(N_T0 * np.exp(-P_T / N_T0 * T), rel=1e-10)


def test_step_optimiser_finds_parent_then_direct_schedule():
    """The case steps exist for: at 10 MeV the beam makes a long-lived parent B that
    decays into a short-lived DI C; at 20 MeV it makes C directly. For C at EOI the best
    schedule is 10 MeV first (build up B), then 20 MeV, and it beats any constant beam.
    For a DI made only directly, the optimum stays the constant beam."""
    lB, lC, T = 1e-4, 1e-3, 1e5
    D = [[0, 0, 0], [0, -lB, 0], [0, lB, -lC]]
    P = [[0, 1e10, 0], [0, 0, 1e10]]                  # rows: 10 MeV, 20 MeV
    prob = _problem(P, D, t_irr=T, di=2)
    best = opt.optimise_steps(prob, 2, "eoi")
    assert best["atoms"] > best["constant"]["atoms"] * 1.05
    assert best["energies"][0] < best["energies"][1]
    direct = _problem([[0, 0, 1e10], [0, 0, 2e10]], D, t_irr=T, di=2)
    best = opt.optimise_steps(direct, 3, "eoi")
    assert best["atoms"] == pytest.approx(best["constant"]["atoms"], rel=1e-6)
    assert best["constant"]["E"] == 20.0


def test_decay_library_parses_modes(tmp_path):
    """MF8/MT457 sections: half-life, NC energies skipped, NDK modes with daughter state
    (RFS) and branching ratio; multi-particle RTYP 1.5 = beta- then n; fission = loss;
    T1/2 = 0 means stable."""
    def line(vals, ns):
        cells = "".join(f"{f'{v:.5E}'.replace('E', ''):>11}" if isinstance(v, float) else f"{v:11d}"
                        for v in vals)
        return f"{cells}1139 8457{ns:5d}"
    sec_a = [line([42099.0, 98.0, 0, 0, 0, 4], 1), line([1000.0, 1.0, 0, 0, 6, 0], 2),
             line([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3), line([0.5, 1.0, 0, 0, 18, 3], 4),
             line([1.0, 1.0, 1.0, 0.0, 0.7, 0.0], 5), line([1.5, 0.0, 1.0, 0.0, 0.2, 0.0], 6),
             line([6.0, 0.0, 1.0, 0.0, 0.1, 0.0], 7)]
    sec_b = [line([42100.0, 99.0, 0, 0, 1, 0], 1), line([0.0, 0.0, 0, 0, 6, 0], 2),
             line([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3), line([0.0, 1.0, 0, 0, 6, 0], 4),
             line([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5)]
    d = tmp_path / "files" / "decay"
    d.mkdir(parents=True)
    (d / "z042").write_text("\n".join(sec_a + sec_b) + "\n")
    lib = opt.DecayLibrary(str(tmp_path))
    lam, br = lib.get((42, 99, 0))
    assert lam == pytest.approx(np.log(2) / 1000.0)
    assert br == [((43, 99, 1), 0.7), ((43, 98, 0), 0.2), (None, 0.1)]
    assert lib.get((42, 100, 0)) == (0.0, [])
    assert lib.get((42, 101, 0)) is None


# ---------------------------------------------------------------- with the library

ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
needs_lib = pytest.mark.skipif(not (ISOTOPIA / "files").is_dir(), reason="ISOTOPIA_DIR not set")


def _paths():
    st = opt.State()
    opt.machine(st)
    return st


def _proton_lines(sym, A, ebeam, thickness=0.05):
    """Optimizer-style input: ISOTOPIA solves Eback from the thickness [cm]."""
    return ["projectile p", f"element {sym}", f"mass {A}", "Ibeam 1", "Tirrad 1 d", "Tcool 1 d",
            f"Ebeam {ebeam}", f"thickness {thickness}"]


@needs_lib
def test_mo100_route_mass_and_precursor_exclusion(tmp_path):
    """p + Mo-100 at 20 MeV makes Tc-99m. Its mass must equal (activity / lambda)
    * M / N_A from the same run; listing Mo99 (a beta- precursor) must exclude the
    route; ISOTOPIA's own Leff must match the requested thickness."""
    paths = _paths()
    t = opt.Target(42, 100, True)
    lines = _proton_lines("Mo", 100, 20.0)
    di = (43, 99, 1)
    M = opt.molar_mass(paths.path, 43, 99)
    assert abs(M - 99) / 99 < 0.01                    # sanity: close to the mass number
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, None, lines, molar_mass=M))
    assert row["status"] == "ok" and row["m_EOC_g"] > 0
    assert row["Leff_cm"] == pytest.approx(0.05, rel=1e-3)
    assert row["stopping"] == "srim"                  # Mo (Z = 42) has a SRIM table
    st = opt.run(lines, out=io.StringIO(), outdir=str(tmp_path))
    key = (43, 99, 1)
    N_eoc = float(st.activity[key][opt.NUMTIME]) / float(st.lam[key])
    assert row["m_EOC_g"] == pytest.approx(N_eoc * M / float(opt.AVOGADRO), rel=1e-5)
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, None, lines, {(42, 99, None)}, M))
    assert row["status"] == "excluded" and "Mo99" in row["note"]


@needs_lib
def test_step_decay_library_real_files():
    """Real JEFF data: every unstable state's branching ratios sum to ~1, and Mo-99
    decays to Tc-99 ground and isomer only. No values are hard-coded."""
    lib = opt.DecayLibrary(str(_paths().path))
    lam, br = lib.get((42, 99, 0))
    assert lam > 0 and {d for d, _ in br} == {(43, 99, 0), (43, 99, 1)}
    for Z in (42, 43):
        for (A, I), (lam, branches) in lib._load(Z).items():
            if lam > 0 and branches:
                assert sum(b for _, b in branches) == pytest.approx(1.0, abs=2e-3), (Z, A, I)


@needs_lib
def test_step_solver_matches_isotopia_for_unfed_products(tmp_path):
    """With one constant step, a product that nothing decays into must have exactly
    ISOTOPIA's inventory at EOI (same production, burnup and decay; float32 in
    ISOTOPIA). Checks every such product of p + Mo-100 at 20 MeV."""
    paths = _paths()
    lines = _proton_lines("Mo", 100, 20.0)
    st = opt.run(lines, out=io.StringIO(), outdir=str(tmp_path))
    target = (42, 100, 0)
    P, P_T = opt.production_states(st, target)
    lib = opt.DecayLibrary(paths.path)
    checked = 0
    for s in P:
        if st.Nisomer[(s[0], s[1])] != -1 or lib.get(s) is None:
            continue
        nodes = opt.build_network(target, s, set(P), lib, [])
        if nodes != [target, s]:
            continue                                   # fed by decay: ISOTOPIA's model differs
        prob = opt.StepProblem(np.array([19.0, 21.0]), np.array([[0.0, P[s]]] * 2), np.array([P_T, P_T]),
                               opt.decay_matrix(nodes, lib), float(st.N_0), float(st.Tir), 0.0, 1)
        ours = prob.evolve([20.0], [float(st.Tir)])[0][1]
        assert ours == pytest.approx(float(st.Niso[(s[0], s[1], -1)][st.Ntime]), rel=1e-4), s
        checked += 1
    assert checked >= 3


@needs_lib
def test_stable_di_mass(tmp_path):
    """Stable Ru-100 from p + Ru-101 at 30 MeV: previously rejected (no activity),
    now reported as a mass. A stable product cannot decay, so its mass cannot
    drop during cooling. Analytic check of the rate: at early times N = N_0 R t
    for direct production, so the first-step rate is N_0 R M / N_A per mA h.
    Tolerance 5%: beta+ feeding from Rh-100 adds a small second-order term."""
    paths = _paths()
    t = opt.Target(44, 101, True)
    lines = _proton_lines("Ru", 101, 30.0)
    M = opt.molar_mass(paths.path, 44, 100)
    row = opt.run_one(opt.RunSpec((44, 100, 0), 2, t, 30.0, None, lines, molar_mass=M))
    assert row["status"] == "ok" and row["m_EOI_g"] > 0
    assert row["m_EOC_g"] >= row["m_EOI_g"] * (1 - 1e-6)
    st = opt.run(lines, out=io.StringIO(), outdir=str(tmp_path))
    key = opt.di_key(st, (44, 100, 0))
    direct = float(st.N_0) * float(st.reaction_rate[key]) * M / float(opt.AVOGADRO)   # g/s
    assert row["m_rate"] == pytest.approx(direct * 3600 / (float(st.Ibeam) * 1e3), rel=0.05)


@needs_lib
def test_beam_that_stops_in_the_target_is_flagged():
    """p + Mo-100 at 8 MeV stops well inside 0.05 cm. ISOTOPIA then irradiates only
    the depth the beam reaches, so Leff < thickness, and the row carries the note
    that main() turns into a warning."""
    paths = _paths()
    M = opt.molar_mass(paths.path, 43, 99)
    row = opt.run_one(opt.RunSpec((43, 99, 1), 2, opt.Target(42, 100, True), 8.0, None,
                                  _proton_lines("Mo", 100, 8.0), molar_mass=M))
    assert row["status"] == "ok" and row["note"].startswith(opt.BEAM_STOPS)
    assert 0 < row["Leff_cm"] < 0.05 and row["Eback_MeV"] < 8.0


# ---------------------------------------------------------------- stopping-power fallback

def test_stopping_warnings_one_per_bethe_target():
    """Only targets that ran on Bethe-Bloch are warned about, once each, in order."""
    rows = [{"projectile": "p", "target": "Np237", "stopping": "bethe"},
            {"projectile": "p", "target": "Mo100", "stopping": "srim"},
            {"projectile": "p", "target": "Np237", "stopping": "bethe"},
            {"projectile": "a", "target": "Np237", "stopping": "bethe"},
            {"projectile": "n", "target": "Mo100", "stopping": None},
            {"projectile": "p", "target": "U238", "status": "error"}]   # failed run: no model
    w = opt.stopping_warnings(rows)
    assert len(w) == 2 and w[0].startswith("p+Np237:") and w[1].startswith("a+Np237:")
    assert all("Bethe-Bloch" in x for x in w)


@needs_lib
def test_target_above_uranium_falls_back_to_bethe_bloch():
    """Np (Z = 93) has no SRIM table, so ISOTOPIA solves Eback with Bethe-Bloch. The
    run must still honour the requested thickness, record the model, and be warned
    about. Check value: Leff equals the requested 0.05 cm, as in the SRIM test above."""
    paths = _paths()
    M = opt.molar_mass(paths.path, 94, 237)
    row = opt.run_one(opt.RunSpec((94, 237, 0), 2, opt.Target(93, 237, False), 30.0, None,
                                  _proton_lines("Np", 237, 30.0), molar_mass=M))
    assert row["status"] in ("ok", "not_produced"), row["note"]
    assert row["stopping"] == "bethe"
    assert row["Leff_cm"] == pytest.approx(0.05, rel=1e-3)
    assert 0 < row["Eback_MeV"] < 30.0
    assert opt.stopping_warnings([row])[0].startswith("p+Np237:")


@needs_lib
def test_bethe_bloch_beam_that_stops_is_an_error():
    """With Bethe-Bloch a beam that would stop in the target cannot be followed (the
    formula is not valid below its maximum), so the run fails with ISOTOPIA's message
    rather than returning a silently wrong Eback. main() lists error rows under Warnings."""
    paths = _paths()
    M = opt.molar_mass(paths.path, 94, 237)
    row = opt.run_one(opt.RunSpec((94, 237, 0), 2, opt.Target(93, 237, False), 8.0, None,
                                  _proton_lines("Np", 237, 8.0), molar_mass=M))
    assert row["status"] == "error" and "Bethe-Bloch" in row["note"]
