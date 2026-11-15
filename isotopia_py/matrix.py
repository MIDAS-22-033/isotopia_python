"""General Bateman solver: dN/dt = M N, solved as N(t) = exp(M t) N(0).

Stage 2: radioactive decay plus reactions on the target. Daughters are not
hit yet. The irradiation itself (two phases, data from an ISOTOPIA run) is
set up in ``irradiation.py``; this module holds the network and the solver.

Physics (manual Eq. 4.1):

    dN_i/dt = sum_k (lambda_k->i + R_k->i) N_k - (lambda_i + R_i) N_i

All nuclides are stacked into one vector N (atoms). The network keeps two
matrices of the same shape, both in s^-1:

* A, decay: -lambda_i on the diagonal and lambda_k * BR_k->i in row i, column k;
* R, reactions: -R_k (the burnup rate) on the diagonal of each irradiated
  nuclide k and R_k->i in row i, column k.

During irradiation M = A + R; during cooling M = A. With M constant, the
exact solution at any time t is the matrix exponential N(t) = exp(M t) N(0)
(manual Section 4.1). Each requested time is computed directly from N(0), so
the time grid only chooses output points: there is no step size and no error
that builds up from one point to the next.

Compared with the analytic port (kinetics.py / equations.f90) this treats
every decay branch, isomeric transitions and decay chains of any length,
which removes defects 1-4 in claude/equations_f90_findings.md.

Numerics: nuclides are ordered so that every product comes before what
makes it (daughters before parents, reaction products before the target).
M is then upper triangular, and scipy.sparse.linalg.expm detects that and
uses Al-Mohy & Higham (2009), Code Fragment 2.1, which recomputes the
diagonal exactly while squaring. Matrices are kept dense (numpy arrays):
for networks of a few hundred nuclides this was 60x faster than sparse
storage in scipy's expm, with identical results.

Limitations:
* Inventories much smaller than the largest one can lose relative accuracy;
  below about 1e-15 of the largest they are rounding noise.
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


@dataclass(frozen=True)
class Reactions:
    """Reactions of one irradiated nuclide k.

    Attributes
    ----------
    loss : float
        Burnup rate R_k [s^-1]: fraction of k's atoms reacting per second.
    products : mapping of Nuclide to float
        Production rate R_k->i [s^-1] of each product i per atom of k.
    """

    loss: float
    products: Mapping[Nuclide, float]


@dataclass
class DecayChain:
    """A set of nuclides linked by decay (and reactions) and their matrices.

    Attributes
    ----------
    nuclides : list of Nuclide
        Keys (Z, A, I), products before the nuclides that make them.
    A : numpy.ndarray
        Decay matrix [s^-1], shape (n, n): dN/dt = A @ N without a beam.
    lam : numpy.ndarray
        Total decay constant of each nuclide [s^-1].
    warnings : list of str
        Truncated chains, untracked fission and data inconsistencies.
    R : numpy.ndarray
        Reaction matrix [s^-1], shape (n, n); all zero without reactions.
        During irradiation dN/dt = (A + R) @ N.
    """

    nuclides: list[Nuclide]
    A: np.ndarray
    lam: np.ndarray
    warnings: list[str] = field(default_factory=list)
    R: np.ndarray | None = None

    def __post_init__(self):
        if self.R is None:
            self.R = np.zeros_like(self.A)

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
# Building the network
# ----------------------------------------------------------------------------

def build_chain(initial: Iterable[Nuclide], library: DecayLibrary,
                on_missing: str = "error",
                reactions: Mapping[Nuclide, Reactions] | None = None) -> DecayChain:
    """Collect every nuclide reachable by decay or reactions and build A and R.

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
    reactions : mapping of Nuclide to Reactions, optional
        Burnup and product rates [s^-1] of each irradiated nuclide.

    Returns
    -------
    DecayChain
        Nuclides with products before their sources, the matrices A and R
        [s^-1] and warnings.
    """
    if on_missing not in ("error", "warn"):
        raise ValueError("on_missing must be 'error' or 'warn'")
    reactions = reactions or {}
    order: list[Nuclide] = []          # finished nuclides, products first
    data = {}                          # decay data per nuclide (None if missing)
    in_progress: set[Nuclide] = set()  # nuclides on the current search path
    warnings: list[str] = []
    cyclic = False

    # Depth-first search: a nuclide is appended only after everything it can
    # turn into (decay daughters and reaction products), so every product
    # ends up before its source in ``order``.
    def visit(n: Nuclide) -> None:
        nonlocal cyclic
        if n in data:
            if n in in_progress:
                cyclic = True
            return
        d = library.get(n)
        data[n] = d
        in_progress.add(n)
        if d is None:
            msg = f"no decay data for {label(n)} in {library.file_for(n[0])}"
            if on_missing == "error":
                raise DecayDataError(msg)
            warnings.append(f"Truncated chain: {msg}; its decay is not followed.")
        else:
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
        if n in reactions:
            for p in reactions[n].products:
                visit(tuple(p))
        in_progress.discard(n)
        order.append(n)

    for n in initial:
        visit(tuple(n))
    if cyclic:
        warnings.append("The decay and reaction links contain a cycle; the matrix is not "
                        "triangular and the less accurate general expm path is used.")

    # Decay matrix: -lambda_i on the diagonal, lambda_k * BR into each daughter row.
    index = {n: k for k, n in enumerate(order)}
    lam = np.array([data[n].lam if data[n] is not None else 0.0 for n in order])
    A = np.diag(-lam)
    for k, n in enumerate(order):
        for m in (data[n].modes if data[n] is not None else ()):
            if m.daughter is not None:
                A[index[m.daughter], k] += lam[k] * m.branching

    # Reaction matrix: -R_k on the diagonal, R_k->i into each product row.
    R = np.zeros_like(A)
    for n, rx in reactions.items():
        k = index[tuple(n)]
        R[k, k] -= rx.loss
        for p, rate in rx.products.items():
            R[index[tuple(p)], k] += rate
    return DecayChain(order, A, lam, warnings, R)


# ----------------------------------------------------------------------------
# Solving
# ----------------------------------------------------------------------------

def propagate(M: np.ndarray, N0: np.ndarray, times: Iterable[float]) -> np.ndarray:
    """N(t) = exp(M t) N0 at each time, for a constant matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        Rate matrix [s^-1], shape (n, n).
    N0 : numpy.ndarray
        Atoms of each nuclide at t = 0, shape (n,).
    times : iterable of float
        Times after t = 0 [s], each >= 0.

    Returns
    -------
    numpy.ndarray
        Atoms, shape (n_times, n).
    """
    times = np.asarray(list(times), dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be >= 0")
    N0 = np.asarray(N0, dtype=float)
    # Each time is an independent exact evaluation, not a step from the previous one.
    return np.array([expm(M * t) @ N0 if t > 0 else N0.copy() for t in times])


def solve(chain: DecayChain, N0: np.ndarray, times: Iterable[float]) -> np.ndarray:
    """Inventories under decay only, N(t) = exp(A t) N0, at each time.

    Parameters
    ----------
    chain : DecayChain
        Nuclides and decay matrix A [s^-1].
    N0 : numpy.ndarray
        Atoms of each nuclide at t = 0, in ``chain.nuclides`` order.
    times : iterable of float
        Times after t = 0 [s], each >= 0.

    Returns
    -------
    numpy.ndarray
        Atoms, shape (n_times, n_nuclides).
    """
    return propagate(chain.A, N0, times)


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
