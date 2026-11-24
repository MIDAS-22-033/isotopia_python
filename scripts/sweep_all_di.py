"""Sweep every projectile, library target and beam energy once and tabulate every product.

Each ISOTOPIA run computes the inventory of every residual product at once, so one
pass over (projectile, target, beam energy) answers the optimal-route question for
every possible desired isotope (DI): every product of every run is a DI candidate.
Runs go through the on-disk cache (isotopia_py.run_cache), so a stopped sweep
restarts where it left off, and optimal_route_v4_step_beam.py searches with the
same settings read the same cache.

    python scripts/sweep_all_di.py --de 1 --jobs 8                      # full sweep
    python scripts/sweep_all_di.py --de 1 --estimate 60                 # time estimate only
    python scripts/sweep_all_di.py -p p --targets Mo100 Cu65 --de 5     # a subset

Outputs (in --out, default runs/sweep_all_di):
    full_table.csv.gz   one row per run and product: all inputs and outputs (gzip)
    best_per_route.csv  each DI, projectile and target at its best beam energy
    best_per_di.csv     each DI's best route
    runs.csv            one row per run: inputs, Eback, Leff, stopping model, status,
                        number of products and the run's ISOTOPIA warnings
    summary.txt         settings, assumptions, counts, run time, warnings
    estimate.txt        (--estimate only) measured time per run and the projection

Needs ISOTOPIA_DIR (and ISOTOPIA_LIBS), exactly as for `isotopia_py.run`.
"""
from __future__ import annotations

import argparse
import csv
import functools
import gzip
import io
import math
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimal_route_v4_step_beam as opt

from isotopia_py import run_cache
from isotopia_py.constants import PARSYM
from isotopia_py.inputdeck import machine
from isotopia_py.state import IsotopiaError, State

RUN_FIELDS = ["projectile", "target", "target_stable", "Ebeam_MeV", "Eback_MeV", "Leff_cm",
              "thickness_cm", "Ibeam_mA", "Tirrad", "Tcool", "stopping"]
PRODUCT_FIELDS = ["product", "Z", "A", "state", "half_life_s"]
OUTPUT_FIELDS = ["m_rate", "m_rate_unit", "m_EOI_g", "m_EOC_g", "A_EOI_Bq", "A_EOC_Bq"]
FULL_FIELDS = RUN_FIELDS + PRODUCT_FIELDS + OUTPUT_FIELDS
BEST_FIELDS = PRODUCT_FIELDS + RUN_FIELDS + OUTPUT_FIELDS
RUNS_FIELDS = RUN_FIELDS + ["status", "from_cache", "n_products", "note", "isotopia_warnings"]


@dataclass(frozen=True)
class Settings:
    """Inputs shared by every run, written into each row (picklable for the workers)."""
    path: str              # ISOTOPIA code folder (files/), for molar masses
    thickness: float       # cm
    ibeam: float           # mA
    tirrad: str            # ISOTOPIA time syntax, e.g. "1 d"
    tcool: str


# ----------------------------------------------------------------------------
# One run: every product with its inputs and outputs
# ----------------------------------------------------------------------------

@functools.cache
def _molar_mass(path: str, Z: int, A: int) -> float | None:
    """g/mol from the decay library (opt.molar_mass), or None if it has no record."""
    try:
        return opt.molar_mass(path, Z, A)
    except LookupError:
        return None


def products(st, target: opt.Target):
    """(ISOTOPIA key, state I) of every product with a nonzero atom count.

    A nuclide without isomers is stored by ISOTOPIA as 'total' (is = -1) and is
    reported as the ground state (I = 0); one with isomers is reported per state
    (0, 1, 2), never as the total, so nothing is counted twice. The target's own
    ground state is the unreacted target, not a product, and is left out.
    """
    states_by_za: dict[tuple[int, int], set[int]] = {}
    for Z, A, is_ in list(st.Niso.keys()):
        if Z >= 1:
            states_by_za.setdefault((Z, A), set()).add(is_)
    for Z, A in sorted(states_by_za):
        n_isomer = st.Nisomer[(Z, A)]
        states = [(-1, 0)] if n_isomer == -1 else [(i, i) for i in range(n_isomer + 1)]
        for is_, I in states:
            if (Z, A, I) == (target.Z, target.A, 0) or not np.any(st.Niso[(Z, A, is_)]):
                continue
            yield (Z, A, is_), I


def run_inputs(spec: opt.RunSpec, s: Settings) -> dict:
    """Input columns of a run (Eback, Leff and stopping are filled from the run)."""
    t = spec.target
    return {"projectile": PARSYM[spec.k0], "target": t.name, "target_stable": t.stable,
            "Ebeam_MeV": spec.ebeam, "Eback_MeV": None, "Leff_cm": None,
            "thickness_cm": s.thickness, "Ibeam_mA": s.ibeam if spec.k0 > 1 else None,
            "Tirrad": s.tirrad, "Tcool": s.tcool, "stopping": None}


def product_rows(st, spec: opt.RunSpec, s: Settings, inputs: dict) -> tuple[list[dict], set]:
    """One row per product: inputs, nuclide, half-life and outputs.

    Masses and mass rate as in the single-DI search (opt.mass_outputs); activities
    are ISOTOPIA's own A = lam N [Bq] (kinetics.activities, manual Sec. 4) at end of
    irradiation and end of cooling; half_life_s = ln 2 / lam with lam from the decay
    library as ISOTOPIA read it (blank: stable or no decay data).
    Returns the rows and the (Z, A) without a molar mass in files/decay.
    """
    rows, no_mass = [], set()
    i_eoc = opt.eoc_index(st)
    for key, I in products(st, spec.target):
        Z, A, _ = key
        M = _molar_mass(s.path, Z, A)
        if M is None:
            no_mass.add((Z, A))
            m_rate = m_eoi = m_eoc = None
        else:
            m_rate, m_eoi, m_eoc = opt.mass_outputs(st, key, M)
        lam = float(st.lam[key])
        act = st.activity[key]
        rows.append({**inputs, "product": opt.nuclide_name(Z, A, I), "Z": Z, "A": A, "state": I,
                     "half_life_s": math.log(2) / lam if lam > 0 else None,
                     "m_rate": m_rate, "m_rate_unit": opt.rate_unit(spec.k0), "m_EOI_g": m_eoi,
                     "m_EOC_g": m_eoc, "A_EOI_Bq": float(act[st.Ntime]), "A_EOC_Bq": float(act[i_eoc])})
    return rows, no_mass


def sweep_one(spec: opt.RunSpec, s: Settings) -> tuple[dict, list[dict], set]:
    """Run (or read from the cache) one case; never raises.

    Returns the runs.csv row, the product rows and the (Z, A) missing a molar mass.
    """
    inputs = run_inputs(spec, s)
    info = {**inputs, "status": "ok", "from_cache": False, "n_products": 0, "note": "",
            "isotopia_warnings": ""}
    try:
        st = run_cache.cached_run(spec.lines, spec.cache_dir)
    except (IsotopiaError, OSError, ValueError) as exc:
        info.update(status="error", note=" ".join(str(exc).split()))
        return info, [], set()
    if spec.k0 > 1:
        inputs.update(Eback_MeV=float(st.Eback), Leff_cm=float(st.Leff), stopping=st.stopping_model)
        info.update(inputs, note=opt.stop_note(st))
    rows, no_mass = product_rows(st, spec, s, inputs)
    info.update(from_cache=isinstance(st, run_cache.RunSnapshot), n_products=len(rows),
                isotopia_warnings=" | ".join(getattr(st, "run_warnings", [])))
    return info, rows, no_mass


# ----------------------------------------------------------------------------
# Building the sweep
# ----------------------------------------------------------------------------

def build_sweep(args, path: str, crosspath: str, warnings: list[str]) -> list[opt.RunSpec]:
    """Every (projectile, library target, energy) run, as opt.RunSpec with di = None."""
    grid = opt.energy_grid(args.emin, args.emax, args.de)
    only = {opt.parse_nuclide(n)[:2] for n in args.targets} if args.targets else None
    specs: list[opt.RunSpec] = []
    for p in dict.fromkeys(args.projectile):
        k0 = PARSYM.index(p)
        targets = opt.library_targets(k0, path, crosspath, args.include_radioactive, warnings)
        if only is not None:
            targets = [t for t in targets if (t.Z, t.A) in only]
        s, _ = opt.build_specs(None, k0, targets, grid, args, set(), warnings)
        for spec in s:
            spec.cache_dir = None if args.no_cache else args.cache
        specs += s
    return specs


# ----------------------------------------------------------------------------
# Time estimate (--estimate N)
# ----------------------------------------------------------------------------

def estimate(specs: list[opt.RunSpec], s: Settings, n: int, seed: int, jobs: int,
             cache_on: bool) -> str:
    """Time a sample of runs afresh and project the whole sweep.

    The sample is stratified by projectile (at least one run each), because run
    times differ by projectile. Each sampled run is timed in this process, without
    the cache, together with building its product rows; the saved snapshot and the
    gzipped table rows are measured too. Runs already in the cache are not counted
    as work left. Wall time with --jobs J assumes J free CPU cores (time / J).
    """
    rng = random.Random(seed)
    by_k0: dict[int, list[opt.RunSpec]] = {}
    for spec in specs:
        by_k0.setdefault(spec.k0, []).append(spec)
    todo = {k0: sum(1 for sp in group if not (cache_on and _in_cache(sp))) for k0, group in by_k0.items()}
    lines, cpu_total, disk_cache, disk_table, rows_total = [], 0.0, 0.0, 0.0, 0.0
    for k0, group in sorted(by_k0.items()):
        k = min(len(group), max(1, round(n * len(group) / len(specs))))
        times, cache_b, table_b, nrows, errors = [], [], [], [], 0
        for spec in rng.sample(group, k):
            t0 = time.perf_counter()
            try:
                st = run_cache._fresh(spec.lines)
            except (IsotopiaError, OSError, ValueError):
                times.append(time.perf_counter() - t0)
                errors += 1
                continue
            rows, _ = product_rows(st, spec, s, run_inputs(spec, s))
            times.append(time.perf_counter() - t0)
            buf = io.BytesIO()
            np.savez_compressed(buf, **run_cache.snapshot_arrays(st))
            cache_b.append(buf.getbuffer().nbytes)
            text = io.StringIO()
            w = csv.DictWriter(text, fieldnames=FULL_FIELDS)
            for r in rows:
                w.writerow({f: opt._fmt(r.get(f)) for f in FULL_FIELDS})
            table_b.append(len(gzip.compress(text.getvalue().encode())))
            nrows.append(len(rows))
        mean = float(np.mean(times))
        cpu_total += mean * todo[k0]
        disk_cache += (float(np.mean(cache_b)) if cache_b else 0.0) * todo[k0]
        disk_table += (float(np.mean(table_b)) if table_b else 0.0) * len(group)
        rows_total += (float(np.mean(nrows)) if nrows else 0.0) * len(group)
        lines.append(f"  {PARSYM[k0]}: {len(group):7d} runs ({todo[k0]} not cached), sampled {k}: "
                     f"mean {mean:.2f} s, median {np.median(times):.2f} s, max {max(times):.2f} s, "
                     f"{np.mean(nrows) if nrows else 0:.0f} products/run, {errors} failed")
    return "\n".join([
        "Time estimate for the sweep (measured on this machine)",
        f"Runs: {len(specs)} in total, {sum(todo.values())} still to compute",
        *lines,
        f"CPU time: {_hms(cpu_total)}",
        f"Wall time: {_hms(cpu_total / jobs)} with --jobs {jobs}"
        + ("" if jobs > 1 else f"; e.g. {_hms(cpu_total / 8)} with --jobs 8 on 8 free cores"),
        f"Cache to add: {_size(disk_cache)}   full_table.csv.gz: {_size(disk_table)}, {rows_total:.3g} rows",
        "Assumes the sampled runs are typical and that --jobs J runs on J free cores."])


def _in_cache(spec: opt.RunSpec) -> bool:
    key = run_cache.cache_key(spec.lines)
    return (Path(spec.cache_dir) / key[:2] / f"{key}.npz").exists()


def _size(nbytes: float) -> str:
    return f"{nbytes / 1e9:.3g} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.3g} MB"


def _hms(seconds: float) -> str:
    h, rem = divmod(round(seconds), 3600)
    return f"{h} h {rem // 60:02d} min" if h else f"{rem // 60} min {rem % 60:02d} s"


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

ASSUMPTIONS = """\
Assumptions and limitations
- One ISOTOPIA run per projectile, ground-state library target (radioactive ones
  only with --include-radioactive) and beam energy; every product with a nonzero
  atom count is a row. Isomeric targets are not run (the input deck cannot take
  them). Products are limited to ISOTOPIA's residual window (Z = Zcomp+1 ..
  Zcomp-10, A = Acomp .. Acomp-20; reacinitial.f90): a DI missing from a target's
  rows is outside that window, not a zero cross section.
- Masses m = N M / N_A from ISOTOPIA's atom counts, with M = AWR m_n from the
  nuclide's record in files/decay/zZZZ; blank if the record is missing (listed
  under Warnings). The mass rate is the first-step slope of N, per mA h for
  charged particles and per g h of target for n and g (not comparable across
  those groups). Activities are ISOTOPIA's A = lam N [Bq] (kinetics.activities);
  half_life_s = ln 2 / lam, blank for stable nuclides or missing decay data.
- Masses and activities are of each product alone: isotopic and chemical purity
  are not reported, and the exclusion list of the single-DI search is not applied.
- Charged particles: the thickness is passed to ISOTOPIA, which solves Eback with
  SRIM tables where they exist (Z = 1-92) and Bethe-Bloch otherwise (column
  'stopping'). Beams that stop in a target irradiate only the depth reached
  (note in runs.csv); with Bethe-Bloch such runs fail (status 'error').
  Photons: whole-MeV energies >= 1 only. Neutrons: one reactor-spectrum run per
  target, no energy scan.
- Double hitting is on (isotopia_py default); nuclides without cross sections are
  not hit, and runs that report missing cross sections say so in runs.csv
  ('isotopia_warnings') and are counted under Warnings.
- Best energy per route and best route per DI are picked on the grid only (no
  refinement between grid energies), by the --metric chosen.
- Runs are memoized (isotopia_py.run_cache), keyed on the input lines, data
  folders and isotopia_py source; delete the cache after editing data files.
- Radiotoxicity and processing time are not part of the ranking yet.
"""


def _write_csv(path: Path, fields: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: opt._fmt(r.get(f)) for f in fields})


def _warning_groups(run_rows: list[dict]) -> list[str]:
    """ISOTOPIA warnings counted by kind (numbers and nuclide lists removed)."""
    counts: dict[str, int] = {}
    for r in run_rows:
        for msg in filter(None, r["isotopia_warnings"].split(" | ")):
            kind = re.sub(r"\d[\d.eE+-]*", "#", msg.split(":", 2)[1] if msg.count(":") >= 1 else msg)
            kind = " ".join(kind.split()[:12])
            counts[kind] = counts.get(kind, 0) + 1
    return [f"{n} runs: ISOTOPIA-warning:{k} ... (full text in runs.csv)" for k, n in
            sorted(counts.items(), key=lambda x: -x[1])]


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-p", "--projectile", nargs="+", choices=PARSYM, default=list(PARSYM),
                    help="projectile(s) (default all: g n p d t h a)")
    ap.add_argument("--targets", nargs="+", default=None, metavar="NAME",
                    help="only these library targets, e.g. Mo100 Cu65 (default all)")
    ap.add_argument("--emin", type=float, default=1.0, help="lowest beam energy [MeV] (default 1)")
    ap.add_argument("--emax", type=float, default=200.0, help="highest beam energy [MeV] (default 200)")
    ap.add_argument("--de", type=float, required=True, help="energy step [MeV]")
    ap.add_argument("--thickness", type=float, default=0.05, help="target thickness [cm] (default 0.05)")
    ap.add_argument("--ibeam", type=float, default=1.0, help="beam current [mA] (default 1)")
    ap.add_argument("--tirrad", default="1 d", help="irradiation time, ISOTOPIA syntax (default '1 d')")
    ap.add_argument("--tcool", default="1 d", help="cooling time, ISOTOPIA syntax (default '1 d')")
    ap.add_argument("--flux", type=float, default=None, help="neutron flux [cm^-2 s^-1] (n only)")
    ap.add_argument("--include-radioactive", action="store_true", help="also run radioactive targets")
    ap.add_argument("--metric", choices=list(opt.METRICS), default="eoc",
                    help="best energy and route by: eoc (mass at end of cooling, default), eoi, rate")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "sweep_all_di"),
                    help="output directory (default runs/sweep_all_di)")
    ap.add_argument("--cache", default=str(ROOT / "runs" / "cache"), help="run cache (default runs/cache)")
    ap.add_argument("--no-cache", action="store_true", help="run every case afresh and save nothing")
    ap.add_argument("--estimate", type=int, default=None, metavar="N",
                    help="only time N sampled runs and project the full sweep's time and disk use")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the --estimate sample")
    args = ap.parse_args(argv)
    if not 0 < args.emin < args.emax <= 200.0:
        ap.error("need 0 < emin < emax <= 200 MeV (library upper limit)")
    if args.de <= 0 or args.thickness <= 0 or args.jobs < 1:
        ap.error("--de, --thickness and --jobs must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = State()
    machine(paths)                                       # resolves ISOTOPIA_DIR / ISOTOPIA_LIBS
    warnings: list[str] = []
    specs = build_sweep(args, paths.path, paths.crosspath, warnings)
    s = Settings(paths.path, args.thickness, args.ibeam, args.tirrad, args.tcool)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.estimate is not None:
        text = estimate(specs, s, args.estimate, args.seed, args.jobs, not args.no_cache)
        (outdir / "estimate.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    # Run every case (cache hits are read back), streaming products to the full table
    col = opt.METRICS[args.metric][0]
    best_route: dict[tuple[str, str, str], dict] = {}
    run_rows, no_mass = [], set()
    n_rows, t0 = 0, time.perf_counter()
    step = max(1, len(specs) // 100)
    worker = functools.partial(sweep_one, s=s)
    with gzip.open(outdir / "full_table.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FULL_FIELDS)
        w.writeheader()
        pool = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
        results = pool.map(worker, specs, chunksize=4) if pool else map(worker, specs)
        try:
            for i, (info, rows, missing) in enumerate(results, 1):
                run_rows.append(info)
                no_mass |= missing
                for r in rows:
                    w.writerow({f: opt._fmt(r.get(f)) for f in FULL_FIELDS})
                    v = r[col]
                    k = (r["product"], r["projectile"], r["target"])
                    if v is not None and v > 0 and (k not in best_route or v > best_route[k][col]):
                        best_route[k] = r
                n_rows += len(rows)
                if i % step == 0 or i == len(specs):
                    el = time.perf_counter() - t0
                    print(f"{i}/{len(specs)} runs, {el:.0f} s, about {el / i * (len(specs) - i):.0f} s left",
                          file=sys.stderr)
        finally:
            if pool:
                pool.shutdown()
    wall = time.perf_counter() - t0

    # Best energy per (DI, projectile, target), and best route per DI
    def order(r):
        return (r["Z"], r["A"], r["state"])
    routes = sorted(best_route.values(), key=lambda r: (order(r), -r[col]))
    best_di: dict[str, dict] = {}
    for r in routes:
        if r["product"] not in best_di:                  # routes are sorted best first per DI
            best_di[r["product"]] = r
    _write_csv(outdir / "best_per_route.csv", BEST_FIELDS, routes)
    _write_csv(outdir / "best_per_di.csv", BEST_FIELDS, best_di.values())
    _write_csv(outdir / "runs.csv", RUNS_FIELDS, run_rows)

    # Warnings: library limits, failed runs, missing data
    errors = [r for r in run_rows if r["status"] == "error"]
    stops = [r for r in run_rows if r["note"].startswith(opt.BEAM_STOPS)]
    if errors:
        warnings.append(f"{len(errors)} runs failed (status 'error' in runs.csv, with ISOTOPIA's message)")
    if stops:
        warnings.append(f"{len(stops)} runs: beam stops in the target; only the depth reached is irradiated")
    bethe = sorted({(r["projectile"], r["target"]) for r in run_rows if r["stopping"] == "bethe"})
    if bethe:
        warnings.append(f"{len(bethe)} charged-particle targets have no SRIM table and use Bethe-Bloch: "
                        + ", ".join(f"{p}+{t}" for p, t in bethe))
    if no_mass:
        warnings.append("no record in files/decay/zZZZ, so masses are blank for: "
                        + ", ".join(opt.nuclide_name(Z, A, None) for Z, A in sorted(no_mass)))
    if args.metric == "rate" and len({p in "pdtha" for p in args.projectile}) > 1:
        warnings.append("--metric rate mixes g/(mA h) (charged particles) with g/(g h) (n, g); "
                        "best_per_di across those groups is not meaningful")
    warnings += _warning_groups(run_rows)

    hits = sum(r["from_cache"] for r in run_rows)
    summary = [
        "Sweep of every projectile, library target and beam energy (all products as DIs)", "",
        (f"Projectiles: {' '.join(dict.fromkeys(args.projectile))}   targets: "
         f"{' '.join(args.targets) if args.targets else 'all in the library'}   grid: {args.emin}-{args.emax} "
         f"MeV, step {args.de} MeV"),
        (f"thickness: {args.thickness} cm   Ibeam: {args.ibeam} mA   Tirrad: {args.tirrad}   "
         f"Tcool: {args.tcool}   best by: {opt.METRICS[args.metric][1]}"),
        f"Cache: {'off' if args.no_cache else args.cache}", "", ASSUMPTIONS,
        f"Runs: {len(run_rows)} ({hits} from the cache, {len(run_rows) - hits} computed, {len(errors)} failed)",
        f"Product rows: {n_rows}   DIs produced: {len(best_di)}   routes: {len(routes)}",
        f"Wall time: {_hms(wall)} with --jobs {args.jobs}", "", "Warnings",
        *[f"- {x}" for x in dict.fromkeys(warnings)]]
    (outdir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[-len(warnings) - 4:]))
    print(f"results in {outdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
