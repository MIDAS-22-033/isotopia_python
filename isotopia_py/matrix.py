"""General Bateman solver: dN/dt = A N, solved as N(t) = exp(A t) N(0).

Stage 1: radioactive decay only. No nuclear reactions, so no irradiation yet.

Physics (manual Eq. 4.1 with every reaction constant R set to zero):

    dN_i/dt = sum_k lambda_k->i N_k - lambda_i N_i

All nuclides are stacked into one vector N (atoms). The decay matrix A (s^-1)
has -lambda_i on the diagonal and lambda_k * BR_k->i in row i, column k. With
A constant, the exact solution at any time t is the matrix exponential
N(t) = exp(A t) N(0) (manual Section 4.1). Each requested time is computed
independently from N(0), so the time grid is only a choice of output points:
there is no step size and no error that builds up from one point to the next.

Compared with the analytic port (kinetics.py / equations.f90) this treats
every branch, isomeric transitions and decay chains of any length, which
removes defects 1-4 in claude/equations_f90_findings.md for decay.

Numerics: the chain is ordered so that every daughter comes before its
parents, which makes A upper triangular. scipy.sparse.linalg.expm detects
that and uses Al-Mohy & Higham (2009), Code Fragment 2.1, which recomputes
the diagonal and first superdiagonal exactly while squaring. That keeps the
absolute error near 1e-15 of the largest inventory, even for very short- or
very long-lived nuclides and for nearly equal decay constants.

Limitations:
* Inventories smaller than about 1e-15 of the largest one are rounding noise.
* Spontaneous fission removes atoms without adding fission products.
* Library branching ratios are used as given. Where they do not sum to 1,
  atoms are gained or lost; a warning names each such nuclide.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import expm

from .decaylib import DecayDataError, DecayLibrary, Nuclide, label

# Report a nuclide whose library branching ratios sum differently from 1 by more than this.
BR_SUM_TOL = 1e-3


@dataclass
class DecayChain:
    """The set of nuclides reachable by decay and their decay matrix.

    Attributes
    ----------
    nuclides : list of Nuclide
        Keys (Z, A, I), daughters before parents.
    A : numpy.ndarray
        Decay matrix [s^-1], shape (n, n): dN/dt = A @ N.
    lam : numpy.ndarray
        Total decay constant of each nuclide [s^-1].
    warnings : list of str
        Truncated chains, untracked fission and data inconsistencies.
    """

    nuclides: list[Nuclide]
    A: np.ndarray
    lam: np.ndarray
    warnings: list[str] = field(default_factory=list)

    @property
    def index(self) -> dict[Nuclide, int]:
        """Position of each nuclide in the vector."""
        return {n: k for k, n in enumerate(self.nuclides)}


@dataclass
class Inventory:
    """Atoms of every nuclide in a chain at a set of times.

    Attributes
    ----------
    times : numpy.ndarray
        Times [s], shape (n_times,).
    nuclides : list of Nuclide
        Keys (Z, A, I), in the column order of ``atoms``.
    atoms : numpy.ndarray
        Number of atoms, shape (n_times, n_nuclides).
    lam : numpy.ndarray
        Decay constants [s^-1], shape (n_nuclides,).
    warnings : list of str
        Warnings carried over from building the chain.
    """

    times: np.ndarray
    nuclides: list[Nuclide]
    atoms: np.ndarray
    lam: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, nuclide: Nuclide) -> np.ndarray:
        """Atoms of one nuclide at every time; zeros if it is not in the chain."""
        try:
            return self.atoms[:, self.nuclides.index(nuclide)]
        except ValueError:
            return np.zeros(len(self.times))

    def activity(self, nuclide: Nuclide) -> np.ndarray:
        """Activity A = lambda N of one nuclide at every time [Bq] (manual Eq. 4.34)."""
        if nuclide not in self.nuclides:
            return np.zeros(len(self.times))
        return self.lam[self.nuclides.index(nuclide)] * self[nuclide]


# ----------------------------------------------------------------------------
# Building the chain
# ----------------------------------------------------------------------------

def build_chain(initial: Iterable[Nuclide], library: DecayLibrary,
                on_missing: str = "error") -> DecayChain:
    """Collect every nuclide reachable by decay and build the decay matrix.

    Parameters
    ----------
    initial : iterable of Nuclide
        Starting nuclides (Z, A, I).
    library : DecayLibrary
        Source of half-lives and decay modes.
    on_missing : {'error', 'warn'}
        What to do when a nuclide has no library entry. 'error' raises
        DecayDataError naming the file and nuclide. 'warn' keeps the nuclide
        as an end point of the chain (its atoms accumulate, decay not followed)
        and records a warning.

    Returns
    -------
    DecayChain
        Nuclides with daughters before parents, the matrix A [s^-1] and warnings.
    """
    if on_missing not in ("error", "warn"):
        raise ValueError("on_missing must be 'error' or 'warn'")
    order: list[Nuclide] = []          # finished nuclides, daughters first
    data = {}                          # decay data per nuclide (None if missing)
    in_progress: set[Nuclide] = set()  # nuclides on the current search path
    warnings: list[str] = []
    cyclic = False

    # Depth-first search: a nuclide is appended only after all its descendants,
    # so every daughter ends up before its parents in ``order``.
    def visit(n: Nuclide) -> None:
        nonlocal cyclic
        if n in data:
            if n in in_progress:
                cyclic = True
            return
        d = library.get(n)
        data[n] = d
        if d is None:
            msg = f"no decay data for {label(n)} in {library.file_for(n[0])}"
            if on_missing == "error":
                raise DecayDataError(msg)
            warnings.append(f"Truncated chain: {msg}; its decay is not followed.")
        else:
            in_progress.add(n)
            br_sum = sum(m.branching for m in d.modes)
            if d.modes and abs(br_sum - 1.0) > BR_SUM_TOL:
                warnings.append(f"Branching ratios of {label(n)} sum to {br_sum:.6g} "
                                f"in {d.source}; used as given.")
            for m in d.modes:
                if m.daughter is None:
                    warnings.append(f"{label(n)}: {m.branching:.3g} of decays are spontaneous "
                                    "fission; fission products are not tracked.")
                else:
                    visit(m.daughter)
            in_progress.discard(n)
        order.append(n)

    for n in initial:
        visit(tuple(n))
    if cyclic:
        warnings.append("Decay data contain a cycle; the matrix is not triangular and "
                        "the less accurate general expm path is used.")

    # Fill the matrix: -lambda_i on the diagonal, lambda_k * BR into each daughter row.
    index = {n: k for k, n in enumerate(order)}
    lam = np.array([data[n].lam if data[n] is not None else 0.0 for n in order])
    A = np.diag(-lam)
    for k, n in enumerate(order):
        for m in (data[n].modes if data[n] is not None else ()):
            if m.daughter is not None:
                A[index[m.daughter], k] += lam[k] * m.branching
    return DecayChain(order, A, lam, warnings)


# ----------------------------------------------------------------------------
# Solving
# ----------------------------------------------------------------------------

def solve(chain: DecayChain, N0: np.ndarray, times: Iterable[float]) -> np.ndarray:
    """Inventories N(t) = exp(A t) N0 at each time.

    Parameters
    ----------
    chain : DecayChain
        Nuclides and decay matrix [s^-1].
    N0 : numpy.ndarray
        Atoms of each nuclide at t = 0, in ``chain.nuclides`` order.
    times : iterable of float
        Times after t = 0 [s], each >= 0.

    Returns
    -------
    numpy.ndarray
        Atoms, shape (n_times, n_nuclides).
    """
    times = np.asarray(list(times), dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be >= 0")
    N0 = np.asarray(N0, dtype=float)
    # Each time is an independent exact evaluation, not a step from the previous one.
    return np.array([expm(chain.A * t) @ N0 if t > 0 else N0.copy() for t in times])


def decay(initial: Mapping[Nuclide, float], times: Iterable[float],
          library: DecayLibrary | None = None, on_missing: str = "error") -> Inventory:
    """Decay an initial inventory and sample it at the given times.

    Parameters
    ----------
    initial : mapping of Nuclide to float
        Atoms of each starting nuclide (Z, A, I) at t = 0.
    times : iterable of float
        Output times [s] after t = 0.
    library : DecayLibrary, optional
        Decay data. Default: the ISOTOPIA decay library (see decaylib).
    on_missing : {'error', 'warn'}
        Handling of nuclides absent from the library; see ``build_chain``.

    Returns
    -------
    Inventory
        Atoms of every nuclide in the chain at each time, with warnings.
    """
    library = library or DecayLibrary()
    chain = build_chain(initial, library, on_missing)
    index = chain.index
    N0 = np.zeros(len(chain.nuclides))
    for n, atoms in initial.items():
        N0[index[tuple(n)]] += atoms
    times = np.asarray(list(times), dtype=float)
    return Inventory(times, chain.nuclides, solve(chain, N0, times), chain.lam, chain.warnings)
