"""On-disk cache of isotopia_py runs (memoization across searches).

One ISOTOPIA run at a given projectile, target, beam energy, thickness and times
computes the inventory of every residual product at once, so the result does not
depend on which desired isotope (DI) is being searched for. ``cached_run`` saves
the parts of each run that the optimal-route search reads (a ``RunSnapshot``);
later calls with the same input, for any DI, read them back instead of running
ISOTOPIA again.

Cache key (SHA-256) of:
  - the input lines exactly as passed to ``run`` (order kept, as ISOTOPIA reads
    them in order),
  - the resolved data folders (ISOTOPIA ``files/`` and ``isotopia.libs``),
  - the source of every isotopia_py module, so any code change starts afresh,
  - ``CACHE_FORMAT``, raised whenever the snapshot layout changes.
Not in the key: the contents of the data files. After updating isotopia.libs or
files/, delete the cache folder.

A run that fails with IsotopiaError is cached as its message, because the same
input fails the same way. Other errors (OSError, ValueError) are not cached.

Files are <cache_dir>/<key[:2]>/<key>.npz. Each is written under a temporary name
and then renamed, so parallel workers never read a half-written file, and loaded
with allow_pickle=False, so a cache file cannot run code.
"""

from __future__ import annotations

import functools
import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .inputdeck import data_directories
from .run import run
from .state import Defaulted, IsotopiaError, State, TimeArrays

CACHE_FORMAT = 2  # 2: activity, lam and ISOTOPIA warnings added

# Scalar State attributes kept in a snapshot (units as in State: MeV, cm, mA, g, s, atoms)
_SCALARS = ("k0", "Eback", "Leff", "Ibeam", "M_target", "N_0", "Tir", "Tco", "Ntime")


# ----------------------------------------------------------------------------
# Snapshot: the State attributes the optimal-route search reads
# ----------------------------------------------------------------------------


class RunSnapshot:
    """Read-only stand-in for a finished ``State``, rebuilt from a cache file.

    Holds the same attribute names and container types as ``State``, so code that
    reads ``st.Niso[key][i]``, ``st.reaction_rate[key]`` or ``st.Nisomer[(Z, A)]``
    gives identical values (float32, as ISOTOPIA computes them) for a cached run
    and a fresh one. Missing keys return the same defaults as ``State``.

    Attributes: k0, Eback [MeV], Leff [cm], Ibeam [mA], M_target [g], N_0 [atoms],
    Tir, Tco [s], Ntime, Tgrid [s], stopping_model, beam_stops, Niso [atoms over
    time], activity [Bq over time, lam * N from kinetics.activities], lam [s^-1],
    reaction_rate [s^-1 per target atom], rpexist, rtyp, Nisomer, run_warnings
    (the run's ISOTOPIA-warning messages, see isotopia_warnings).
    """

    def __init__(self, arrays: dict[str, np.ndarray]):
        for name in _SCALARS:
            setattr(self, name, arrays[name][()])  # [()] keeps the numpy scalar type
        self.Tgrid = arrays["Tgrid"]
        self.stopping_model = str(arrays["stopping_model"])
        self.beam_stops = bool(arrays["beam_stops"])
        self.Niso = _time_arrays(arrays["niso_keys"], arrays["niso"])
        self.activity = _time_arrays(arrays["act_keys"], arrays["act"])
        self.lam = _defaulted(np.float32(0), arrays["lam_keys"], arrays["lam"])
        self.reaction_rate = _defaulted(np.float32(0), arrays["rr_keys"], arrays["rr"])
        self.rpexist = _defaulted(False, arrays["rp_keys"], arrays["rp"].astype(bool))
        self.rtyp = _defaulted(0, arrays["rtyp_keys"], arrays["rtyp"])
        self.Nisomer = _defaulted(-1, arrays["niso_za"], arrays["nisomer"])
        self.run_warnings = [str(w) for w in arrays["warnings"]]


def _time_arrays(keys: np.ndarray, rows: np.ndarray) -> TimeArrays:
    """TimeArrays from an (n, 3) key array and an (n, NUMTIME + 1) array of rows."""
    return TimeArrays(zip(map(tuple, keys.tolist()), rows))


def _split_time(d: TimeArrays) -> tuple[np.ndarray, np.ndarray]:
    """TimeArrays -> ((n, 3) keys, (n, NUMTIME + 1) float32 rows)."""
    keys = list(d.keys())
    rows = np.array([d[k] for k in keys], dtype=np.float32).reshape(len(keys), -1)
    return np.array(keys, dtype=np.int64).reshape(-1, 3), rows


def _defaulted(default, keys: np.ndarray, values: np.ndarray) -> Defaulted:
    """Defaulted dict from an (n, k) integer key array and n values."""
    d = Defaulted(default)
    d.update(
        zip(
            map(tuple, keys.tolist()),
            values.tolist() if values.dtype == bool else values,
        )
    )
    return d


def _split(d: dict, width: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    """dict {(int, ...): value} -> (keys array of shape (n, width), values array)."""
    keys = np.array(list(d.keys()), dtype=np.int64).reshape(-1, width)
    return keys, np.array(list(d.values()), dtype=dtype)


def snapshot_arrays(st: State) -> dict[str, np.ndarray]:
    """The arrays saved for one run (see RunSnapshot for what each holds)."""
    out = {name: np.asarray(getattr(st, name)) for name in _SCALARS}
    out["Tgrid"] = np.asarray(st.Tgrid)
    out["stopping_model"] = np.asarray(getattr(st, "stopping_model", ""))
    out["beam_stops"] = np.asarray(bool(getattr(st, "beam_stops", False)))
    out["niso_keys"], out["niso"] = _split_time(st.Niso)
    out["act_keys"], out["act"] = _split_time(st.activity)
    out["lam_keys"], out["lam"] = _split(st.lam, 3, np.float32)
    out["rr_keys"], out["rr"] = _split(st.reaction_rate, 3, np.float32)
    out["rp_keys"], out["rp"] = _split(st.rpexist, 3, bool)
    out["rtyp_keys"], out["rtyp"] = _split(st.rtyp, 3, np.int64)
    out["niso_za"], out["nisomer"] = _split(st.Nisomer, 2, np.int64)
    out["warnings"] = np.array(getattr(st, "run_warnings", []), dtype=str)
    return out


def isotopia_warnings(text: str) -> list[str]:
    """ISOTOPIA-warning messages in a run's output, each joined with its indented
    detail lines (e.g. the list of nuclides with missing cross sections)."""
    messages: list[list[str]] = []
    current = None
    for line in text.splitlines():
        if "ISOTOPIA-warning" in line:
            current = [line.strip()]
            messages.append(current)
        elif current is not None and line.startswith("   ") and line.strip():
            current.append(line.strip())
        else:
            current = None
    return list(dict.fromkeys(" ".join(" ".join(m).split()) for m in messages))


# ----------------------------------------------------------------------------
# Cache key and file I/O
# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """SHA-256 of every isotopia_py source file, computed once per process."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def cache_key(lines: list[str]) -> str:
    """Hex key for one run: input lines, data folders, code version, format."""
    code_dir, libs = data_directories()
    text = "\n".join(
        [
            f"format {CACHE_FORMAT}",
            code_fingerprint(),
            os.path.realpath(code_dir),
            os.path.realpath(libs),
            *lines,
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write arrays to path atomically (temporary file in the same folder, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read(path: Path) -> dict[str, np.ndarray] | None:
    """Arrays from a cache file, or None if it is missing or unreadable (rerun then)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def cached_run(
    lines: list[str], cache_dir: str | os.PathLike | None
) -> State | RunSnapshot:
    """Run isotopia_py on `lines`, or read the result of an identical earlier run.

    Parameters
    ----------
    lines : list of str
        ISOTOPIA input keywords (manual Sec. 6), as for ``isotopia_py.run``.
    cache_dir : path or None
        Cache folder; None runs without the cache.

    Returns
    -------
    State or RunSnapshot
        The fresh State on a cache miss, else the snapshot; both carry the
        attributes listed in RunSnapshot.

    Raises
    ------
    IsotopiaError
        If the run fails, fresh or as recorded in the cache.
    """
    if cache_dir is None:
        return _fresh(lines)
    path = Path(cache_dir) / (key := cache_key(lines))[:2] / f"{key}.npz"
    arrays = _read(path) if path.exists() else None
    if arrays is not None:
        if "error" in arrays:
            raise IsotopiaError(str(arrays["error"]))
        return RunSnapshot(arrays)
    try:
        st = _fresh(lines)
    except IsotopiaError as exc:
        _write(path, {"error": np.asarray(str(exc))})
        raise
    _write(path, snapshot_arrays(st))
    return st


def _fresh(lines: list[str]) -> State:
    """One isotopia_py run in a throwaway output folder; its ISOTOPIA-warning
    messages are kept as ``st.run_warnings``."""
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        st = run(lines, out=out, outdir=tmp)
    st.run_warnings = isotopia_warnings(out.getvalue())
    return st
