"""
Cost Integration Module

Combines fuzzy logic congestion scores and Bayesian road-risk probabilities
into a single effective edge cost for the A* routing algorithm.

Pipeline:
    Existing road data (traffic_factor, base_distance)
    → Fuzzy Logic → congestion_score (0–100) → congestion_factor (1.0–3.0)
    → Bayesian Reasoning → risk_probability (0.0–1.0) → risk_factor (≥1.0)
    → effective_cost = base_distance × congestion_factor × risk_factor

When both features are DISABLED, the formula reduces to the original:
    effective_cost = base_distance × traffic_factor

Design Decisions:
    - The fuzzy congestion factor REPLACES the raw traffic_factor rather than
      multiplying on top of it, to avoid double-counting congestion.
    - The Bayesian risk factor is an ADDITIONAL multiplier on top of congestion,
      since it captures a different dimension of risk (blockage probability).
    - Confirmed road closures (if implemented) should be handled at the graph
      level by removing edges, not by setting high costs.
"""

from typing import Optional, Dict, Any

from algorithms.fuzzy_logic import compute_fuzzy_congestion, congestion_score_to_factor
from algorithms.bayesian_reasoning import compute_bayesian_risk, risk_probability_to_factor


def compute_effective_weight(
    base_distance: float,
    traffic_factor: float,
    fuzzy_enabled: bool = False,
    bayesian_enabled: bool = False,
    bayesian_evidence: Optional[Dict[str, Any]] = None,
    bayesian_penalty_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Compute the effective edge weight integrating fuzzy and Bayesian adjustments.

    This is the single integration function called by CityGraph.get_neighbors_enhanced().
    It preserves exact backward compatibility when both features are disabled.

    Args:
        base_distance: Euclidean distance between the two nodes.
        traffic_factor: Existing traffic multiplier (1.0–3.0).
        fuzzy_enabled: Whether to apply fuzzy congestion adjustment.
        bayesian_enabled: Whether to apply Bayesian risk adjustment.
        bayesian_evidence: Dict with optional keys 'rain' (bool), 'accident' (bool),
            'traffic_level' (str). Missing keys are treated as unknown evidence.
        bayesian_penalty_weight: Maximum cost penalty at P(blocked)=1.0.

    Returns:
        Dict with:
            'weight': The effective edge cost (float, ≥ 0).
            'congestion_score': Fuzzy congestion score (0–100), or None if disabled.
            'congestion_factor': Cost multiplier from fuzzy logic, or traffic_factor.
            'risk_probability': Bayesian P(blocked), or None if disabled.
            'risk_factor': Cost multiplier from Bayesian reasoning, or 1.0.
    """
    if bayesian_evidence is None:
        bayesian_evidence = {}

    # ── Congestion Factor ──
    congestion_score = None
    if fuzzy_enabled:
        congestion_score = compute_fuzzy_congestion(traffic_factor)
        congestion_factor = congestion_score_to_factor(congestion_score)
    else:
        # Original behavior: use raw traffic_factor
        congestion_factor = traffic_factor

    # ── Risk Factor ──
    risk_probability = None
    risk_factor = 1.0
    if bayesian_enabled:
        risk_probability = compute_bayesian_risk(
            traffic_factor=traffic_factor,
            rain=bayesian_evidence.get('rain'),
            accident=bayesian_evidence.get('accident'),
            traffic_level=bayesian_evidence.get('traffic_level'),
        )
        risk_factor = risk_probability_to_factor(risk_probability, bayesian_penalty_weight)

    # ── Effective Weight ──
    weight = base_distance * congestion_factor * risk_factor

    # Safety: ensure non-negative cost
    weight = max(0.0, weight)

    return {
        'weight': weight,
        'congestion_score': congestion_score,
        'congestion_factor': congestion_factor,
        'risk_probability': risk_probability,
        'risk_factor': risk_factor,
    }
