"""Energy loss of a charged particle through a layer of given thickness (new; not in the Fortran).

ISOTOPIA takes the energies entering and leaving the target, Ebeam and Eback, and returns the
thickness (manual Eq. 4.23, reactionrates.f90 lines 158-169):

    L_eff = integral from Eback to Ebeam of dE / S(E)                     [cm]

with S(E) the stopping power in MeV/cm. This module solves the same relation in both directions,
so that thickness can be an input:

    thickness(E_in, E_out)   = R(E_in) - R(E_out)                         [cm]
    exit_energy(E_in, t)     = the E_out for which R(E_in) - R(E_out) = t [MeV]

where R(E) = integral from E_floor to E of dE'/S(E') is the path length the particle travels
while slowing from E to the lowest tabulated energy E_floor. The two are exact inverses of each
other, so exit_energy(E_in, thickness(E_in, E_out)) returns E_out.

How R(E) is computed
Between neighbouring table energies E_i and E_{i+1} the stopping power is taken as a power law,
S(E) = S_i (E/E_i)^b (a straight line on a log-log plot; the same interpolation as
srim.StoppingTable). For a power law the integral of dE/S has a closed form, so each segment is
integrated exactly, and the result can be inverted exactly. No numerical integration or
root-finding is needed.

Physics model and assumptions
- Continuous slowing down approximation (CSDA): every particle loses energy at the mean rate
  S(E). Energy-loss straggling is ignored, so E_out is the mean exit energy. This is the same
  model as ISOTOPIA's Eq. 4.23. CSDA ranges are defined this way in ICRU Report 49, "Stopping
  Powers and Ranges for Protons and Alpha Particles" (1993), and the NIST PSTAR/ASTAR tables.
- The thickness is the path length along the particle's track. Multiple scattering makes the
  straight-line depth shorter than the path; that is ignored (as in ISOTOPIA). Compared with the
  projected-range column of the SRIM tables (p and a in Ni and Au), the path length is longer
  than the mean depth by 1-4% for a particle stopping from 10 MeV and by 5-37% from 1 MeV. So
  the error is small for layers the beam crosses, and matters when it nearly stops in one.
- The layer is uniform: one material, one density.
- A particle whose path length to E_floor is not longer than the layer is reported as stopped
  (exit energy 0). The last part of the track, below E_floor (1 keV for the SRIM tables), is
  not followed. For protons in Ni that is 66 angstrom (SRIM's projected range at 1 keV); in a
  gas it is longer (136 micron for protons in H2 at 25 degC, 1 atm).
- Energies above the table, or an entry energy below it, raise ValueError; nothing is
  extrapolated.

Used by rates.setup_stopping for three things: Eback when 'thickness' is given for a
charged-particle target, the beam spectrum of runs that use SRIM stopping, and the layers in
front of and behind the target (keyword 'layer').
"""

from __future__ import annotations

import numpy as np

from .srim import StoppingTable


class RangeTable:
    """Path length versus energy for one projectile in one uniform material.

    Parameters
    ----------
    energy : array_like
        Projectile kinetic energies, MeV, strictly increasing (at least two).
    stopping : array_like
        Total stopping power at ``energy``, MeV/cm, positive.
    """

    def __init__(self, energy, stopping):
        E = np.asarray(energy, dtype=float)
        S = np.asarray(stopping, dtype=float)
        if E.ndim != 1 or E.shape != S.shape or E.size < 2:
            raise ValueError(
                "energy and stopping must be 1-D arrays of the same length (>= 2)"
            )
        if np.any(np.diff(E) <= 0) or np.any(E <= 0) or np.any(S <= 0):
            raise ValueError(
                "energies must be positive and increasing, stopping powers positive"
            )
        self.energy, self.stopping = E, S
        # Power-law exponent b of each segment, and k = 1 - b, which appears in the integral
        self._k = 1.0 - np.diff(np.log(S)) / np.diff(np.log(E))
        # R at each table energy: running sum of the exact segment integrals (R = 0 at E_floor)
        segments = _segment_path(E[:-1], S[:-1], self._k, E[1:])
        self._path = np.concatenate(([0.0], np.cumsum(segments)))

    @classmethod
    def from_srim(cls, table: StoppingTable, rho: float, mass: float) -> RangeTable:
        """Build from a SRIM table for a material of given density and atomic mass.

        Parameters
        ----------
        table : srim.StoppingTable
            SRIM stopping per atom (electronic + nuclear are both used).
        rho : float
            Density of the layer, g/cm3.
        mass : float
            Atomic mass of the layer material, g/mol (u).

        Returns
        -------
        RangeTable
        """
        return cls(table.energy, table.stopping_power(table.energy, rho, mass))

    @property
    def emin(self) -> float:
        """Lowest table energy E_floor, MeV."""
        return float(self.energy[0])

    @property
    def emax(self) -> float:
        """Highest table energy, MeV."""
        return float(self.energy[-1])

    def _check(self, E, what: str) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        if np.any(E < self.emin) or np.any(E > self.emax):
            raise ValueError(
                f"{what} outside the stopping table ({self.emin:g} to {self.emax:g} MeV); "
                "no extrapolation is done"
            )
        return E

    def path_length(self, E) -> np.ndarray:
        """Path length R(E) travelled while slowing from E to the table floor.

        Parameters
        ----------
        E : float or array_like
            Projectile kinetic energy, MeV, within the table.

        Returns
        -------
        numpy.ndarray
            R(E), cm. Add the (untracked) path below the floor to get the full CSDA range.
        """
        E = self._check(E, "energy")
        # Segment that holds each E: index i with energy[i] <= E < energy[i+1]
        i = np.clip(
            np.searchsorted(self.energy, E, side="right") - 1, 0, self.energy.size - 2
        )
        return self._path[i] + _segment_path(
            self.energy[i], self.stopping[i], self._k[i], E
        )

    def thickness(self, E_in, E_out) -> np.ndarray:
        """Layer thickness that slows the particle from E_in to E_out (manual Eq. 4.23).

        Parameters
        ----------
        E_in, E_out : float or array_like
            Energies entering and leaving the layer, MeV, with E_out <= E_in.

        Returns
        -------
        numpy.ndarray
            Thickness (path length), cm.
        """
        E_in, E_out = self._check(E_in, "E_in"), self._check(E_out, "E_out")
        if np.any(E_out > E_in):
            raise ValueError("E_out must not exceed E_in")
        return self.path_length(E_in) - self.path_length(E_out)

    def exit_energy(self, E_in, thickness) -> np.ndarray:
        """Mean energy leaving a layer of given thickness (inverse of ``thickness``).

        Parameters
        ----------
        E_in : float or array_like
            Energy entering the layer, MeV, within the table.
        thickness : float or array_like
            Layer thickness (path length), cm, >= 0.

        Returns
        -------
        numpy.ndarray
            Exit energy, MeV. 0.0 means the particle stops in the layer: its path length down
            to the table floor is not longer than the layer.
        """
        thickness = np.asarray(thickness, dtype=float)
        if np.any(thickness < 0):
            raise ValueError("thickness must be >= 0")
        # Path length still ahead of the particle when it reaches the back of the layer
        remaining = self.path_length(E_in) - thickness
        stopped = remaining <= 0
        r = np.where(stopped, 0.0, remaining)
        # Segment that holds r, then invert the closed-form segment integral for the energy
        i = np.clip(
            np.searchsorted(self._path, r, side="right") - 1, 0, self.energy.size - 2
        )
        E_i, S_i, k = self.energy[i], self.stopping[i], self._k[i]
        u = (r - self._path[i]) * S_i / E_i
        small = np.abs(k) < 1e-12
        k_safe = np.where(small, 1.0, k)
        E_out = E_i * np.where(small, np.exp(u), np.exp(np.log1p(k_safe * u) / k_safe))
        # Guard against rounding just past the table ends
        return np.where(stopped, 0.0, np.clip(E_out, self.emin, self.emax))


def _segment_path(E_i, S_i, k, E):
    """Integral of dE'/S from E_i to E for the power law S = S_i (E'/E_i)^(1-k).

    Closed form: (E_i/S_i) * ((E/E_i)^k - 1) / k, which tends to (E_i/S_i) * ln(E/E_i) as k -> 0.
    Energies in MeV, stopping in MeV/cm; returns cm.
    """
    log_ratio = np.log(E / E_i)
    small = np.abs(k) < 1e-12
    k_safe = np.where(small, 1.0, k)
    return E_i / S_i * np.where(small, log_ratio, np.expm1(k_safe * log_ratio) / k_safe)
