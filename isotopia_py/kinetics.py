"""Small, transparent Python translations of ISOTOPIA kinetics formulas.

The functions here translate the *direct target -> product* part of
source/equations.f90 and the activity relation in source/activities.f90.
They intentionally do not model parent decay chains, isomers, energy-dependent
cross sections, or target self-shielding.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def target_inventory(
    times_s: Iterable[float],
    initial_atoms: float,
    target_reaction_constant_s: float,
    irradiation_time_s: float,
) -> list[float]:
    """Target atoms vs time, using ISOTOPIA's exponential depletion formula.

    During irradiation: N_target(t) = N0 * exp(-R_target*t).
    During cooling, this simplified model holds the target inventory fixed,
    matching the target branch in equations.f90.
    """
    result = []
    for t in times_s:
        if t < 0:
            raise ValueError("Times must be non-negative.")
        if t <= irradiation_time_s:
            result.append(initial_atoms * math.exp(-target_reaction_constant_s * t))
        else:
            result.append(initial_atoms * math.exp(-target_reaction_constant_s * irradiation_time_s))
    return result


def direct_product_inventory(
    times_s: Iterable[float],
    initial_target_atoms: float,
    target_reaction_constant_s: float,
    product_reaction_constant_s: float,
    product_decay_constant_s: float,
    irradiation_time_s: float,
) -> list[float]:
    """Inventory of a product made directly from the target, then cooled.

    This translates the direct-production terms in equations.f90:

      N_product(t) = N0 * R_product *
          (exp(-R_target*t) - exp(-lambda_product*t)) /
          (lambda_product - R_target)

    for t during irradiation. If the two constants are effectively equal,
    the limiting expression N0*R_product*t*exp(-lambda_product*t) is used.

    After irradiation, this direct-only model applies radioactive decay to
    the end-of-irradiation product inventory. Feeding from radioactive parent
    nuclides is deliberately omitted.
    """
    if irradiation_time_s < 0:
        raise ValueError("Irradiation time must be non-negative.")

    result = []
    rt = target_reaction_constant_s
    ri = product_reaction_constant_s
    lam = product_decay_constant_s
    denom = lam - rt
    eps = 1e-12 * max(abs(lam), abs(rt), 1e-30)

    # Inventory at end of irradiation is used for all cooling-time points.
    if abs(denom) > eps:
        n_at_eoi = (
            initial_target_atoms * ri
            * (math.exp(-rt * irradiation_time_s) - math.exp(-lam * irradiation_time_s))
            / denom
        )
    else:
        n_at_eoi = initial_target_atoms * ri * irradiation_time_s * math.exp(-lam * irradiation_time_s)

    for t in times_s:
        if t < 0:
            raise ValueError("Times must be non-negative.")
        if t <= irradiation_time_s:
            if abs(denom) > eps:
                n = initial_target_atoms * ri * (
                    math.exp(-rt * t) - math.exp(-lam * t)
                ) / denom
            else:
                n = initial_target_atoms * ri * t * math.exp(-lam * t)
        else:
            n = n_at_eoi * math.exp(-lam * (t - irradiation_time_s))
        result.append(max(n, 0.0))
    return result


def activity_bq(inventory_atoms: Iterable[float], decay_constant_s: float) -> list[float]:
    """Activity A=lambda*N in decays per second (Bq)."""
    if decay_constant_s < 0:
        raise ValueError("Decay constant must be non-negative.")
    return [decay_constant_s * n for n in inventory_atoms]
