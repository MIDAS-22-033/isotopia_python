"""Run the Python port on the ISOTOPIA sample cases and compare with references.

    python scripts/run_samples.py                     # all samples vs samples/*/org
    python scripts/run_samples.py p-Mo100-Tc099m      # selected samples
    python scripts/run_samples.py --ref runs/fortran  # vs another set of reference runs
    python scripts/run_samples.py --no-run            # only compare existing results

Needs ISOTOPIA_DIR (the isotopia code directory, for files/ and samples/) and,
unless isotopia.libs sits next to it, ISOTOPIA_LIBS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isotopia_py.compare import compare_dirs, summarize  # noqa: E402
from isotopia_py.run import run  # noqa: E402
from isotopia_py.state import IsotopiaError  # noqa: E402


def isotopia_dir() -> Path:
    env = os.environ.get("ISOTOPIA_DIR")
    return Path(env) if env else ROOT.parent / "isotopia"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="*", help="sample names (default: all)")
    p.add_argument("--ref", help="directory with one reference subdirectory per case "
                                 "(default: the samples' own org/ directories)")
    p.add_argument("--out", default=str(ROOT / "runs" / "python"), help="where to write Python results")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--floor", type=float, default=1e-5)
    p.add_argument("--no-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    samples = isotopia_dir() / "samples"
    cases = args.cases or sorted(d.name for d in samples.iterdir() if (d / "org" / "isotopia.inp").is_file())
    failed = 0
    for case in cases:
        inp = samples / case / "org" / "isotopia.inp"
        outdir = Path(args.out) / case
        if not args.no_run:
            outdir.mkdir(parents=True, exist_ok=True)
            for old in outdir.iterdir():
                old.unlink()
            t0 = time.time()
            with open(outdir / "isotopia.out", "w", encoding="utf-8", newline="\n") as out:
                try:
                    run(inp.read_text(encoding="utf-8").splitlines(), out=out, outdir=str(outdir))
                except IsotopiaError as exc:
                    out.write(str(exc) + "\n")
            dt = time.time() - t0
        else:
            dt = 0.0
        ref = Path(args.ref) / case if args.ref else samples / case / "org"
        results = compare_dirs(outdir, ref, args.rtol, args.floor)
        bad = [r for r in results if r.status != "ok"]
        failed += bool(bad)
        print(f"{case:22s} {dt:5.1f}s  {summarize(results)}")
        for r in results:
            if r.status != "ok" or (args.verbose and not r.identical):
                print(f"    {r.status:8s} {r.name:18s} max_rel={r.max_rel:.2e}  {r.worst}")
                for note in r.notes[:3]:
                    print(f"             {note}")
    print(f"\n{len(cases) - failed}/{len(cases)} cases within rtol={args.rtol:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
