"""
Fuzzy Logic Module for Traffic Congestion Severity Estimation

Uses Mamdani-style fuzzy inference with centroid defuzzification to
estimate a congestion severity score (0–100) from available road data.

Inputs:
    - traffic_factor: The existing traffic multiplier from CityGraph edges
      (1.0 = clear, up to 3.0 = heavy traffic). This is the primary input
      already available in the project.
    - average_speed_ratio: Optional ratio of current speed to free-flow speed
      (0.0–1.0, where 1.0 = free-flow and 0.0 = standstill). This input is
      SIMULATED — the project does not measure actual vehicle speeds. It is
      derived from traffic_factor as: speed_ratio = 1.0 / traffic_factor.

Output:
    - congestion_score: A numerical severity value in [0, 100].

Membership Functions:
    - Traffic density: LOW, MEDIUM, HIGH (triangular)
    - Speed ratio:     FAST, MODERATE, SLOW (triangular)
    - Congestion:      LOW, MEDIUM, HIGH, VERY_HIGH (triangular)

Fuzzy Rules:
    R1: IF traffic_density is LOW                           → congestion is LOW
    R2: IF traffic_density is MEDIUM                        → congestion is MEDIUM
    R3: IF traffic_density is HIGH                          → congestion is HIGH
    R4: IF traffic_density is HIGH AND speed is SLOW        → congestion is VERY_HIGH
    R5: IF traffic_density is MEDIUM AND speed is SLOW      → congestion is HIGH
    R6: IF traffic_density is LOW AND speed is FAST          → congestion is LOW

Note: All probability values are computed from the mathematical definitions
of the membership functions and inference rules. They are deterministic for
identical inputs and model parameters.
"""

import numpy as np
from typing import Optional


# ─── Membership Function Helpers ───

def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Triangular membership function.

    Args:
        x: Input values array.
        a: Left foot of the triangle (membership = 0).
        b: Peak of the triangle (membership = 1).
        c: Right foot of the triangle (membership = 0).

    Returns:
        Membership degrees for each value in x, in [0, 1].
    """
    return np.maximum(
        0.0,
        np.minimum((x - a) / max(b - a, 1e-10), (c - x) / max(c - b, 1e-10))
    )


def _trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Trapezoidal membership function.

    Args:
        x: Input values array.
        a: Left foot (membership = 0).
        b: Left shoulder (membership = 1).
        c: Right shoulder (membership = 1).
        d: Right foot (membership = 0).

    Returns:
        Membership degrees for each value in x, in [0, 1].
    """
    return np.maximum(
        0.0,
        np.minimum(
            np.minimum((x - a) / max(b - a, 1e-10), 1.0),
            (d - x) / max(d - c, 1e-10)
        )
    )


# ─── Input Membership Functions ───

# Traffic density: mapped from traffic_factor (1.0–3.0)
# LOW: peak at 1.0, fades by 1.8
# MEDIUM: peak at 2.0, spans 1.2–2.8
# HIGH: peak at 3.0, rises from 2.2

def _traffic_density_low(x: np.ndarray) -> np.ndarray:
    """LOW traffic density membership (traffic_factor domain 1.0–3.0)."""
    return _trimf(x, 0.5, 1.0, 1.8)


def _traffic_density_medium(x: np.ndarray) -> np.ndarray:
    """MEDIUM traffic density membership (traffic_factor domain 1.0–3.0)."""
    return _trimf(x, 1.2, 2.0, 2.8)


def _traffic_density_high(x: np.ndarray) -> np.ndarray:
    """HIGH traffic density membership (traffic_factor domain 1.0–3.0)."""
    return _trimf(x, 2.2, 3.0, 3.5)


# Speed ratio: mapped from speed_ratio (0.0–1.0)
# SLOW: peak at 0.0, fades by 0.5
# MODERATE: peak at 0.5, spans 0.2–0.8
# FAST: peak at 1.0, rises from 0.5

def _speed_slow(x: np.ndarray) -> np.ndarray:
    """SLOW speed membership (speed_ratio domain 0.0–1.0)."""
    return _trimf(x, -0.3, 0.0, 0.5)


def _speed_moderate(x: np.ndarray) -> np.ndarray:
    """MODERATE speed membership (speed_ratio domain 0.0–1.0)."""
    return _trimf(x, 0.2, 0.5, 0.8)


def _speed_fast(x: np.ndarray) -> np.ndarray:
    """FAST speed membership (speed_ratio domain 0.0–1.0)."""
    return _trimf(x, 0.5, 1.0, 1.3)


# ─── Output Membership Functions (Congestion Severity 0–100) ───

def _congestion_low(x: np.ndarray) -> np.ndarray:
    """LOW congestion output membership."""
    return _trimf(x, 0.0, 10.0, 35.0)


def _congestion_medium(x: np.ndarray) -> np.ndarray:
    """MEDIUM congestion output membership."""
    return _trimf(x, 20.0, 45.0, 65.0)


def _congestion_high(x: np.ndarray) -> np.ndarray:
    """HIGH congestion output membership."""
    return _trimf(x, 50.0, 70.0, 85.0)


def _congestion_very_high(x: np.ndarray) -> np.ndarray:
    """VERY_HIGH congestion output membership."""
    return _trimf(x, 75.0, 90.0, 100.0)


# ─── Centroid Defuzzification ───

def _centroid_defuzzify(output_space: np.ndarray, aggregated: np.ndarray) -> float:
    """
    Centroid defuzzification.

    Args:
        output_space: The discrete output universe (e.g., 0–100).
        aggregated: Aggregated membership values for each point.

    Returns:
        The defuzzified crisp value. Returns 0.0 if the aggregated area is zero.
    """
    total_area = np.sum(aggregated)
    if total_area < 1e-10:
        return 0.0
    return float(np.sum(output_space * aggregated) / total_area)


# ─── Main Fuzzy Inference Function ───

def compute_fuzzy_congestion(
    traffic_factor: float,
    speed_ratio: Optional[float] = None,
) -> float:
    """
    Compute a fuzzy congestion severity score using Mamdani inference.

    This function takes the existing traffic_factor from the road data and
    optionally a speed_ratio, and returns a congestion severity score (0–100).

    Args:
        traffic_factor: The traffic multiplier from CityGraph edges (1.0–3.0).
            Values outside this range are clamped.
        speed_ratio: Optional ratio of current speed to free-flow speed (0.0–1.0).
            If None, it is derived from traffic_factor as 1.0 / traffic_factor.
            This is a SIMULATED value — the project does not have real speed data.

    Returns:
        Congestion severity score in the range [0, 100].
        Deterministic for identical inputs.
    """
    # ── Input validation and clamping ──
    if not isinstance(traffic_factor, (int, float)) or np.isnan(traffic_factor):
        traffic_factor = 1.0  # Safe default: clear road
    traffic_factor = float(np.clip(traffic_factor, 1.0, 3.0))

    if speed_ratio is None:
        # Derive simulated speed ratio from traffic_factor
        speed_ratio = 1.0 / traffic_factor
    if not isinstance(speed_ratio, (int, float)) or np.isnan(speed_ratio):
        speed_ratio = 1.0 / traffic_factor
    speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))

    # ── Fuzzification: compute membership degrees for crisp inputs ──
    tf_arr = np.array([traffic_factor])
    sr_arr = np.array([speed_ratio])

    td_low = float(_traffic_density_low(tf_arr)[0])
    td_med = float(_traffic_density_medium(tf_arr)[0])
    td_high = float(_traffic_density_high(tf_arr)[0])

    sp_slow = float(_speed_slow(sr_arr)[0])
    sp_mod = float(_speed_moderate(sr_arr)[0])
    sp_fast = float(_speed_fast(sr_arr)[0])

    # ── Fuzzy Inference Rules (Mamdani: use min for AND, max for aggregation) ──
    # R1: IF traffic_density is LOW → congestion is LOW
    rule1 = td_low

    # R2: IF traffic_density is MEDIUM → congestion is MEDIUM
    rule2 = td_med

    # R3: IF traffic_density is HIGH → congestion is HIGH
    rule3 = td_high

    # R4: IF traffic_density is HIGH AND speed is SLOW → congestion is VERY_HIGH
    rule4 = min(td_high, sp_slow)

    # R5: IF traffic_density is MEDIUM AND speed is SLOW → congestion is HIGH
    rule5 = min(td_med, sp_slow)

    # R6: IF traffic_density is LOW AND speed is FAST → congestion is LOW
    rule6 = min(td_low, sp_fast)

    # ── Aggregate output activations per output category ──
    activation_low = max(rule1, rule6)
    activation_medium = rule2
    activation_high = max(rule3, rule5)
    activation_very_high = rule4

    # ── Defuzzification: centroid method on discrete output universe ──
    output_space = np.linspace(0, 100, 500)

    # Clip each output membership function by its activation level (Mamdani implication)
    out_low = np.minimum(activation_low, _congestion_low(output_space))
    out_med = np.minimum(activation_medium, _congestion_medium(output_space))
    out_high = np.minimum(activation_high, _congestion_high(output_space))
    out_vhigh = np.minimum(activation_very_high, _congestion_very_high(output_space))

    # Aggregate: max of all clipped output sets
    aggregated = np.maximum(np.maximum(out_low, out_med),
                            np.maximum(out_high, out_vhigh))

    score = _centroid_defuzzify(output_space, aggregated)

    # Ensure output is within [0, 100]
    return float(np.clip(score, 0.0, 100.0))


def congestion_score_to_factor(congestion_score: float) -> float:
    """
    Convert a fuzzy congestion score (0–100) to a cost multiplier.

    This REPLACES the raw traffic_factor in the cost formula to avoid
    double-counting congestion. The mapping is:

        congestion_score 0   → factor 1.0 (no penalty)
        congestion_score 100 → factor 3.0 (maximum penalty, same range as traffic_factor)

    The linear mapping preserves the same cost range as the original system.

    Args:
        congestion_score: Fuzzy congestion severity in [0, 100].

    Returns:
        Cost multiplier in [1.0, 3.0].
    """
    if not isinstance(congestion_score, (int, float)) or np.isnan(congestion_score):
        return 1.0
    score = float(np.clip(congestion_score, 0.0, 100.0))
    return 1.0 + (score / 100.0) * 2.0
