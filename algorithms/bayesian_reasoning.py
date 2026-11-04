"""
Bayesian Reasoning Module for Road Risk Estimation

Uses a small, transparent Bayesian Network to estimate the probability
that a road is congested or blocked, given uncertain evidence.

Model Structure (Bayesian Network):
    Rain ──┐
           ├──→ Road_Blocked
    Accident ─┤
              │
    Traffic_Level ─┘

Target variable: P(Road_Blocked | observed evidence)

Evidence Variables and States:
    - rain:          True / False  (weather condition; SIMULATED — project has
                     no real weather data, default: False)
    - accident:      True / False  (accident reported on road; SIMULATED,
                     default: False)
    - traffic_level: 'low' / 'medium' / 'high'  (derived from existing
                     traffic_factor: low ≤1.5, medium ≤2.2, high >2.2)

Conditional Probability Tables (manually specified, NOT trained on data):
    These CPTs encode reasonable domain assumptions about how weather,
    accidents, and traffic level relate to road blockage. They are NOT
    empirically calibrated — they represent expert-style heuristic priors.

Implementation:
    - Exact inference via enumeration (network is small enough).
    - Missing evidence is handled by marginalizing over unknown variables.
    - Identical inputs always produce identical outputs (deterministic).
    - If a road is confirmed blocked in the existing system, the Bayesian
      posterior does NOT override that authoritative status.
"""

from typing import Optional, Dict, Any
import numpy as np


# ─── Prior Probabilities ───

# P(Rain) — unconditional prior probability of rain
P_RAIN = 0.2

# P(Accident) — unconditional prior probability of an accident on a road
P_ACCIDENT = 0.05


# ─── Conditional Probability Tables ───

# P(Traffic_Level | Rain)
# Rows: rain = {True, False}
# Columns: traffic_level = {low, medium, high}
P_TRAFFIC_GIVEN_RAIN: Dict[bool, Dict[str, float]] = {
    True:  {'low': 0.15, 'medium': 0.35, 'high': 0.50},
    False: {'low': 0.50, 'medium': 0.35, 'high': 0.15},
}

# P(Road_Blocked | Rain, Accident, Traffic_Level)
# This is the main CPT. Road blockage depends on all three parent variables.
# Format: P_BLOCKED[rain][accident][traffic_level] = probability
P_BLOCKED: Dict[bool, Dict[bool, Dict[str, float]]] = {
    # Rain = True
    True: {
        # Rain + Accident
        True: {
            'low':    0.60,
            'medium': 0.75,
            'high':   0.90,
        },
        # Rain + No Accident
        False: {
            'low':    0.05,
            'medium': 0.15,
            'high':   0.35,
        },
    },
    # Rain = False
    False: {
        # No Rain + Accident
        True: {
            'low':    0.40,
            'medium': 0.55,
            'high':   0.70,
        },
        # No Rain + No Accident
        False: {
            'low':    0.01,
            'medium': 0.05,
            'high':   0.15,
        },
    },
}


def _traffic_factor_to_level(traffic_factor: float) -> str:
    """
    Convert a numeric traffic_factor to a discrete traffic level.

    Mapping (consistent with the existing project's color coding):
        traffic_factor ≤ 1.5 → 'low'
        traffic_factor ≤ 2.2 → 'medium'
        traffic_factor > 2.2 → 'high'

    Args:
        traffic_factor: Traffic multiplier (1.0–3.0).

    Returns:
        One of 'low', 'medium', 'high'.
    """
    if traffic_factor <= 1.5:
        return 'low'
    elif traffic_factor <= 2.2:
        return 'medium'
    else:
        return 'high'


def compute_bayesian_risk(
    traffic_factor: float,
    rain: Optional[bool] = None,
    accident: Optional[bool] = None,
    traffic_level: Optional[str] = None,
) -> float:
    """
    Compute the posterior probability P(Road_Blocked | evidence) using
    Bayesian inference via exact enumeration.

    This function is deterministic: identical inputs always produce the
    same output. No randomness is involved.

    Args:
        traffic_factor: The existing traffic multiplier (1.0–3.0).
            Used to derive traffic_level if not explicitly provided.
        rain: Whether rain is occurring. None means unknown (marginalized).
            SIMULATED — the project has no real weather data.
        accident: Whether an accident is reported. None means unknown.
            SIMULATED — the project has no real accident feed.
        traffic_level: Explicit traffic level override ('low'/'medium'/'high').
            If None, derived from traffic_factor.

    Returns:
        Posterior probability of road blockage, in [0.0, 1.0].
    """
    # ── Input validation ──
    if not isinstance(traffic_factor, (int, float)) or np.isnan(traffic_factor):
        traffic_factor = 1.0
    traffic_factor = float(np.clip(traffic_factor, 1.0, 3.0))

    # Derive traffic level from traffic_factor if not explicitly provided
    if traffic_level is None:
        traffic_level = _traffic_factor_to_level(traffic_factor)
    if traffic_level not in ('low', 'medium', 'high'):
        traffic_level = _traffic_factor_to_level(traffic_factor)

    # ── Bayesian Inference via Enumeration ──
    # We compute P(Blocked=True | evidence) by summing over unknown variables.
    #
    # P(Blocked | evidence) = Σ_hidden P(Rain) × P(Accident) ×
    #                          P(Traffic|Rain) × P(Blocked|Rain,Accident,Traffic)
    #                         / P(evidence)
    #
    # Since we normalize at the end, we compute the unnormalized joint for
    # Blocked=True and Blocked=False, then normalize.

    # Determine which values of rain and accident to enumerate
    rain_values = [rain] if rain is not None else [True, False]
    accident_values = [accident] if accident is not None else [True, False]

    prob_blocked = 0.0
    prob_not_blocked = 0.0

    for r in rain_values:
        # P(Rain = r)
        p_rain = P_RAIN if r else (1.0 - P_RAIN)

        for a in accident_values:
            # P(Accident = a)
            p_acc = P_ACCIDENT if a else (1.0 - P_ACCIDENT)

            # P(Traffic_Level = tl | Rain = r)
            p_traffic = P_TRAFFIC_GIVEN_RAIN[r][traffic_level]

            # P(Blocked = True | Rain=r, Accident=a, Traffic=tl)
            p_b = P_BLOCKED[r][a][traffic_level]

            joint_weight = p_rain * p_acc * p_traffic

            prob_blocked += joint_weight * p_b
            prob_not_blocked += joint_weight * (1.0 - p_b)

    # Normalize
    total = prob_blocked + prob_not_blocked
    if total < 1e-10:
        return 0.0

    posterior = prob_blocked / total
    return float(np.clip(posterior, 0.0, 1.0))


def risk_probability_to_factor(
    risk_probability: float,
    penalty_weight: float = 1.0,
) -> float:
    """
    Convert a Bayesian road-risk probability to a cost multiplier.

    Formula: factor = 1.0 + penalty_weight × risk_probability

    This produces:
        risk_probability = 0.0 → factor = 1.0 (no penalty)
        risk_probability = 1.0 → factor = 1.0 + penalty_weight

    Args:
        risk_probability: Posterior P(blocked) in [0.0, 1.0].
        penalty_weight: Maximum additional cost factor at P=1.0.
            Default 1.0 means a fully-blocked road doubles the cost.
            Configurable via UI.

    Returns:
        Cost multiplier ≥ 1.0.
    """
    if not isinstance(risk_probability, (int, float)) or np.isnan(risk_probability):
        return 1.0
    if not isinstance(penalty_weight, (int, float)) or np.isnan(penalty_weight):
        penalty_weight = 1.0

    rp = float(np.clip(risk_probability, 0.0, 1.0))
    pw = max(0.0, float(penalty_weight))

    return 1.0 + pw * rp
