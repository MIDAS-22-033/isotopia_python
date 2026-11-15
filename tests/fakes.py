"""Shared test helpers: an in-memory decay library and access to the sample cases."""
from __future__ import annotations

import io
import math
import os
from pathlib import Path

from isotopia_py.decaylib import DecayMode, NuclideDecay, default_decay_dir

# ISOTOPIA code directory (decay files, samples) and cross-section library, as in test_samples.py
ROOT = Path(__file__).resolve().parents[1]
ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
SAMPLES = ISOTOPIA / "samples"
LIBS = Path(os.environ.get("ISOTOPIA_LIBS") or ISOTOPIA.parent / "isotopia.libs")
DECAY_DIR = default_decay_dir()


def sample_cases() -> list[str]:
    """Names of the ISOTOPIA sample cases that have a reference input."""
    if not SAMPLES.is_dir():
        return []
    return sorted(d.name for d in SAMPLES.iterdir() if (d / "org" / "isotopia.inp").is_file())


def run_sample_capturing(case: str, outdir: Path, monkeypatch, capture) -> None:
    """Run one sample through the port, calling ``capture(st)`` after each kinetics.equations.

    ``kinetics.equations`` runs once per target isotope (twice or more for a
    natural target), so ``capture`` sees every isotope run, before
    ``kinetics.activities`` changes the inventories of isomer totals.
    """
    from isotopia_py import kinetics
    from isotopia_py.run import run

    original = kinetics._equations

    def hooked(st):
        original(st)
        capture(st)

    monkeypatch.setattr(kinetics, "_equations", hooked)
    lines = (SAMPLES / case / "org" / "isotopia.inp").read_text(encoding="utf-8").splitlines()
    run(lines, out=io.StringIO(), outdir=str(outdir))


class FakeLibrary:
    """In-memory stand-in for DecayLibrary: {nuclide: (lambda, [(daughter, BR), ...])}."""

    def __init__(self, table):
        self.table = table

    def file_for(self, z):
        return Path(f"fake/z{z:03d}")

    def get(self, n):
        if n not in self.table:
            return None
        lam, modes = self.table[n]
        half = math.log(2) / lam if lam > 0 else math.inf
        return NuclideDecay(n, half, lam, tuple(DecayMode("1", d, br) for d, br in modes), "fake")
