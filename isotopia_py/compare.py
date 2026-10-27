"""Compare two ISOTOPIA output directories, number by number.

Lines are matched one to one. Non-numeric text must agree exactly; numbers
are compared by relative difference. Header lines that legitimately differ
between runs (user, date, cross-section path) are skipped, as in the Fortran
samples/verify script.

ISOTOPIA works in single precision, so two correct builds (even two Fortran
builds on different platforms) differ in the last digit. Two places amplify
that noise, and are compared accordingly:

* "N years N days N hours N minutes N seconds" strings are compared as total
  seconds, since one float32 step of a 1e10 s time shifts the seconds field.
* In data blocks, a number may also differ by up to ``floor`` times the
  largest value in its column. This covers the production-rate column (a
  finite difference of nearly equal activities) and nuclides whose inventory
  is a tiny difference of large terms.

    python -m isotopia_py.compare NEW_DIR REFERENCE_DIR [--rtol 1e-5] [--floor 1e-5]
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SKIP = re.compile(r"^\s*(#\s*)?(user|date|User|Date):\s|^\s*crosspath\s|Execution time:")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE]?[+-]\d+)?$")
_BREAKDOWN = re.compile(
    r"(\d+)\s*(?:years|y)\s+(\d+)\s*(?:days|d)\s+(\d+)\s*(?:hours|h)\s+(\d+)\s*(?:minutes|m)"
    r"\s+(\d+)\s*(?:seconds|s)\b")
_IGNORE_FILES = {"isotopia.inp", "outputdiff", ".DS_Store"}
_YEAR = 365.25 * 86400.0


def _num(tok: str):
    if not _NUMBER.match(tok):
        return None
    m = re.match(r"^([+-]?[\d.]+)([+-]\d+)$", tok)  # Fortran 1.0+100
    return float(f"{m.group(1)}e{m.group(2)}" if m else tok)


def _seconds(m: re.Match) -> str:
    y, d, h, mi, s = (int(g) for g in m.groups())
    return f" {y * _YEAR + d * 86400.0 + h * 3600.0 + mi * 60.0 + s:.10e}s "


@dataclass
class FileResult:
    name: str
    status: str = "ok"            # ok, missing, extra, text, numeric
    max_rel: float = 0.0
    worst: str = ""
    identical: bool = True
    notes: list[str] = field(default_factory=list)


def _lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [ln for ln in text.splitlines() if not _SKIP.search(ln)]


def _is_data_row(tokens: list[str]) -> bool:
    return len(tokens) >= 2 and all(_num(t) is not None for t in tokens)


def compare_files(new: Path, ref: Path, rtol: float, floor: float) -> FileResult:
    res = FileResult(new.name)
    a, b = _lines(new), _lines(ref)
    if a == b:
        return res
    res.identical = False
    if len(a) != len(b):
        res.status = "text"
        res.notes.append(f"{len(a)} lines vs {len(b)} in reference")
    a = [_BREAKDOWN.sub(_seconds, ln) for ln in a]
    b = [_BREAKDOWN.sub(_seconds, ln) for ln in b]
    # Column maxima of the reference data rows, for the absolute floor
    colmax: dict[tuple[int, int], float] = {}
    for ln in b:
        t = ln.split()
        if _is_data_row(t):
            for c, tok in enumerate(t):
                k = (len(t), c)
                colmax[k] = max(colmax.get(k, 0.0), abs(_num(tok)))
    excess = 0.0
    for k, (la, lb) in enumerate(zip(a, b), start=1):
        if la == lb:
            continue
        ta, tb = la.split(), lb.split()
        if len(ta) != len(tb):
            res.status = "text"
            res.notes.append(f"line {k}: {la.strip()!r} vs {lb.strip()!r}")
            continue
        data = _is_data_row(tb)
        for c, (x, y) in enumerate(zip(ta, tb)):
            if x == y:
                continue
            xs, ys = x.rstrip("s"), y.rstrip("s")
            fx, fy = _num(xs), _num(ys)
            if fx is None or fy is None:
                res.status = "text"
                res.notes.append(f"line {k}: {x!r} vs {y!r}")
                continue
            diff = abs(fx - fy)
            scale = max(abs(fx), abs(fy))
            rel = diff / scale if scale > 0 else 0.0
            allowed = rtol * scale + (floor * colmax.get((len(tb), c), 0.0) if data else 0.0)
            if "." in ys and not re.search(r"[eE]|\d[+-]\d", ys):
                # Fixed-point field: allow one unit in the last printed decimal
                allowed = max(allowed, 1.0001 * 10.0 ** -len(ys.split(".")[1]))
            if diff > allowed and allowed > 0:
                excess = max(excess, diff / allowed)
            if rel > res.max_rel:
                res.max_rel = rel
                res.worst = f"line {k}: {x} vs {y}"
    if res.status == "ok" and excess > 1.0:
        res.status = "numeric"
    return res


def compare_dirs(new: Path, ref: Path, rtol: float = 1e-5, floor: float = 1e-5) -> list[FileResult]:
    new, ref = Path(new), Path(ref)
    names_new = {p.name for p in new.iterdir() if p.is_file()} - _IGNORE_FILES
    names_ref = {p.name for p in ref.iterdir() if p.is_file()} - _IGNORE_FILES
    results = []
    for name in sorted(names_new | names_ref):
        if name not in names_new:
            results.append(FileResult(name, "missing", identical=False))
        elif name not in names_ref:
            results.append(FileResult(name, "extra", identical=False))
        else:
            results.append(compare_files(new / name, ref / name, rtol, floor))
    return results


def summarize(results: list[FileResult]) -> str:
    n = len(results)
    ident = sum(r.identical for r in results)
    bad = [r for r in results if r.status != "ok"]
    s = f"{n} files: {ident} identical, {n - ident - len(bad)} within tolerance, {len(bad)} failing"
    worst = max((r for r in results if r.status == "ok"), key=lambda r: r.max_rel, default=None)
    if worst and worst.max_rel > 0:
        s += f"; largest relative difference {worst.max_rel:.1e} ({worst.name})"
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("new")
    p.add_argument("reference")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--floor", type=float, default=1e-5)
    p.add_argument("-v", "--verbose", action="store_true", help="list every differing file")
    args = p.parse_args(argv)
    results = compare_dirs(Path(args.new), Path(args.reference), args.rtol, args.floor)
    for r in results:
        if r.status != "ok" or (args.verbose and not r.identical):
            print(f"  {r.status:8s} {r.name}  max_rel={r.max_rel:.2e} {r.worst}")
            for note in r.notes[:5]:
                print(f"           {note}")
    print(summarize(results))
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
