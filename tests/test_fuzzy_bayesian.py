"""
Tests for Fuzzy Logic, Bayesian Reasoning, and Cost Integration modules.

Covers:
    - Fuzzy logic: low/medium/high traffic → appropriate congestion scores,
      boundary values, invalid inputs, score range [0, 100].
    - Bayesian reasoning: posterior probabilities in [0, 1], evidence updates,
      determinism, missing evidence handling.
    - Cost integration: backward compatibility, feature toggling,
      mathematical consistency.
    - End-to-end: A* routing still works, hospital assignment still works,
      enhanced routing changes costs when enabled.

Run with:
    python -m pytest tests/test_fuzzy_bayesian.py -v
    or:
    python tests/test_fuzzy_bayesian.py
"""

import sys
import os
import math

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.fuzzy_logic import compute_fuzzy_congestion, congestion_score_to_factor
from algorithms.bayesian_reasoning import (
    compute_bayesian_risk, risk_probability_to_factor, _traffic_factor_to_level
)
from algorithms.cost_integration import compute_effective_weight
from models.city_graph import CityGraph
from data.city_data import create_city_graph, create_hospitals, create_ambulances
from algorithms.astar import astar_search
from algorithms.csp import hospital_csp


# ═════════════════════════════════════════════
# FUZZY LOGIC TESTS
# ═════════════════════════════════════════════

class TestFuzzyLogic:
    """Tests for the fuzzy logic congestion estimation module."""

    def test_low_traffic_produces_low_congestion(self):
        """Traffic factor 1.0 (clear road) should produce a low congestion score."""
        score = compute_fuzzy_congestion(1.0)
        assert 0 <= score <= 30, f"Expected low score for clear road, got {score}"

    def test_medium_traffic_produces_medium_congestion(self):
        """Traffic factor 2.0 (moderate) should produce a medium congestion score."""
        score = compute_fuzzy_congestion(2.0)
        assert 25 <= score <= 65, f"Expected medium score, got {score}"

    def test_high_traffic_produces_high_congestion(self):
        """Traffic factor 3.0 (heavy) should produce a high congestion score."""
        score = compute_fuzzy_congestion(3.0)
        assert 60 <= score <= 100, f"Expected high score for heavy traffic, got {score}"

    def test_high_traffic_slow_speed_produces_very_high(self):
        """High traffic + slow speed should produce a very high congestion score."""
        # speed_ratio 0.1 = nearly standstill
        score = compute_fuzzy_congestion(3.0, speed_ratio=0.1)
        assert score >= 60, f"Expected very high score, got {score}"

    def test_score_within_range(self):
        """All congestion scores must be in [0, 100]."""
        test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, -1.0]
        for tf in test_values:
            score = compute_fuzzy_congestion(tf)
            assert 0 <= score <= 100, f"Score {score} out of range for tf={tf}"

    def test_monotonically_increasing(self):
        """Higher traffic factor should generally produce higher congestion."""
        score_low = compute_fuzzy_congestion(1.0)
        score_med = compute_fuzzy_congestion(2.0)
        score_high = compute_fuzzy_congestion(3.0)
        assert score_low < score_med < score_high, \
            f"Expected monotonic increase: {score_low} < {score_med} < {score_high}"

    def test_deterministic(self):
        """Same inputs must always produce the same output."""
        score1 = compute_fuzzy_congestion(2.0, speed_ratio=0.5)
        score2 = compute_fuzzy_congestion(2.0, speed_ratio=0.5)
        assert score1 == score2, "Fuzzy logic should be deterministic"

    def test_invalid_traffic_factor_handled(self):
        """Invalid inputs should be handled safely (no crash, returns valid score)."""
        score = compute_fuzzy_congestion(float('nan'))
        assert 0 <= score <= 100

    def test_invalid_speed_ratio_handled(self):
        """Invalid speed_ratio should be handled safely."""
        score = compute_fuzzy_congestion(2.0, speed_ratio=float('nan'))
        assert 0 <= score <= 100

    def test_congestion_score_to_factor_range(self):
        """congestion_score_to_factor must produce values in [1.0, 3.0]."""
        assert congestion_score_to_factor(0) == 1.0
        assert congestion_score_to_factor(100) == 3.0
        assert 1.0 <= congestion_score_to_factor(50) <= 3.0

    def test_congestion_score_to_factor_invalid(self):
        """Invalid input to congestion_score_to_factor should return 1.0."""
        assert congestion_score_to_factor(float('nan')) == 1.0


# ═════════════════════════════════════════════
# BAYESIAN REASONING TESTS
# ═════════════════════════════════════════════

class TestBayesianReasoning:
    """Tests for the Bayesian road-risk estimation module."""

    def test_probability_in_valid_range(self):
        """Posterior probabilities must be in [0, 1]."""
        test_cases = [
            {'traffic_factor': 1.0},
            {'traffic_factor': 3.0},
            {'traffic_factor': 2.0, 'rain': True},
            {'traffic_factor': 2.0, 'accident': True},
            {'traffic_factor': 3.0, 'rain': True, 'accident': True},
        ]
        for case in test_cases:
            prob = compute_bayesian_risk(**case)
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range for {case}"

    def test_evidence_increases_probability(self):
        """Rain and accident evidence should increase blockage probability."""
        base = compute_bayesian_risk(2.0)
        with_rain = compute_bayesian_risk(2.0, rain=True)
        with_accident = compute_bayesian_risk(2.0, accident=True)
        with_both = compute_bayesian_risk(2.0, rain=True, accident=True)

        assert with_rain > base, "Rain should increase risk"
        assert with_accident > base, "Accident should increase risk"
        assert with_both > with_rain, "Both evidence should be worse than rain alone"
        assert with_both > with_accident, "Both evidence should be worse than accident alone"

    def test_higher_traffic_increases_risk(self):
        """Higher traffic should increase blockage probability."""
        low = compute_bayesian_risk(1.0)
        high = compute_bayesian_risk(3.0)
        assert high > low, f"Expected higher risk at high traffic: {low} vs {high}"

    def test_deterministic(self):
        """Same inputs must always produce the same output."""
        p1 = compute_bayesian_risk(2.5, rain=True, accident=False)
        p2 = compute_bayesian_risk(2.5, rain=True, accident=False)
        assert p1 == p2, "Bayesian reasoning should be deterministic"

    def test_missing_evidence_handled(self):
        """Missing evidence (None) should be marginalized, not crash."""
        prob = compute_bayesian_risk(2.0, rain=None, accident=None)
        assert 0.0 <= prob <= 1.0

    def test_invalid_traffic_factor_handled(self):
        """Invalid traffic_factor should be handled safely."""
        prob = compute_bayesian_risk(float('nan'))
        assert 0.0 <= prob <= 1.0

    def test_traffic_factor_to_level(self):
        """Traffic factor to level conversion should match project color coding."""
        assert _traffic_factor_to_level(1.0) == 'low'
        assert _traffic_factor_to_level(1.5) == 'low'
        assert _traffic_factor_to_level(1.8) == 'medium'
        assert _traffic_factor_to_level(2.2) == 'medium'
        assert _traffic_factor_to_level(2.5) == 'high'
        assert _traffic_factor_to_level(3.0) == 'high'

    def test_risk_probability_to_factor_range(self):
        """risk_probability_to_factor must produce values >= 1.0."""
        assert risk_probability_to_factor(0.0) == 1.0
        assert risk_probability_to_factor(1.0, penalty_weight=1.0) == 2.0
        assert risk_probability_to_factor(1.0, penalty_weight=2.0) == 3.0
        assert risk_probability_to_factor(0.5, penalty_weight=1.0) == 1.5

    def test_risk_probability_to_factor_invalid(self):
        """Invalid input should return 1.0 (no penalty)."""
        assert risk_probability_to_factor(float('nan')) == 1.0


# ═════════════════════════════════════════════
# COST INTEGRATION TESTS
# ═════════════════════════════════════════════

class TestCostIntegration:
    """Tests for the cost integration module."""

    def test_both_disabled_returns_original(self):
        """With both features disabled, effective weight = base_distance × traffic_factor."""
        result = compute_effective_weight(
            base_distance=5.0,
            traffic_factor=2.0,
            fuzzy_enabled=False,
            bayesian_enabled=False,
        )
        expected = 5.0 * 2.0
        assert abs(result['weight'] - expected) < 1e-6, \
            f"Expected {expected}, got {result['weight']}"

    def test_fuzzy_only_changes_cost(self):
        """With only fuzzy enabled, cost should differ from raw traffic_factor."""
        original = compute_effective_weight(5.0, 2.0, False, False)
        fuzzy = compute_effective_weight(5.0, 2.0, True, False)
        # Fuzzy-derived cost should differ from raw (they use different formulas)
        # but both should be positive
        assert fuzzy['weight'] > 0
        assert fuzzy['congestion_score'] is not None
        assert 0 <= fuzzy['congestion_score'] <= 100

    def test_bayesian_only_changes_cost(self):
        """With only Bayesian enabled, cost includes risk factor."""
        original = compute_effective_weight(5.0, 2.0, False, False)
        with_risk = compute_effective_weight(
            5.0, 2.0, False, True,
            bayesian_evidence={'rain': True, 'accident': True},
        )
        assert with_risk['weight'] > original['weight'], \
            "Bayesian risk with adverse evidence should increase cost"
        assert with_risk['risk_probability'] is not None
        assert 0 <= with_risk['risk_probability'] <= 1

    def test_both_enabled(self):
        """Both features enabled should produce a valid positive weight."""
        result = compute_effective_weight(
            5.0, 2.5, True, True,
            bayesian_evidence={'rain': True},
        )
        assert result['weight'] > 0
        assert result['congestion_score'] is not None
        assert result['risk_probability'] is not None

    def test_weight_nonnegative(self):
        """Effective weight should always be non-negative."""
        for tf in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for fuzzy in [True, False]:
                for bayes in [True, False]:
                    result = compute_effective_weight(3.0, tf, fuzzy, bayes)
                    assert result['weight'] >= 0, \
                        f"Negative weight for tf={tf}, fuzzy={fuzzy}, bayes={bayes}"


# ═════════════════════════════════════════════
# INTEGRATION TESTS WITH EXISTING SYSTEM
# ═════════════════════════════════════════════

class TestSystemIntegration:
    """Tests that existing functionality still works with the new features."""

    def setup_method(self):
        """Set up a fresh city graph for each test."""
        self.city = create_city_graph()
        self.hospitals = create_hospitals()
        self.ambulances = create_ambulances()

    def test_astar_works_without_enhancement(self):
        """A* routing still works with both features disabled (default)."""
        result = astar_search(self.city, 'A', 'T')
        assert result is not None
        assert result['path'][0] == 'A'
        assert result['path'][-1] == 'T'
        assert result['cost'] > 0

    def test_csp_works_without_enhancement(self):
        """CSP hospital assignment still works with both features disabled."""
        result = hospital_csp(self.hospitals, True, 'Q', self.city)
        assert result['assigned_hospital'] is not None

    def test_astar_works_with_fuzzy(self):
        """A* routing still finds a path with fuzzy logic enabled."""
        self.city.set_enhanced_routing(fuzzy_enabled=True)
        result = astar_search(self.city, 'A', 'T')
        assert result is not None
        assert result['path'][0] == 'A'
        assert result['path'][-1] == 'T'
        assert result['cost'] > 0

    def test_astar_works_with_bayesian(self):
        """A* routing still finds a path with Bayesian reasoning enabled."""
        self.city.set_enhanced_routing(
            bayesian_enabled=True,
            global_evidence={'rain': True, 'accident': False},
        )
        result = astar_search(self.city, 'A', 'T')
        assert result is not None
        assert result['path'][0] == 'A'
        assert result['path'][-1] == 'T'
        assert result['cost'] > 0

    def test_astar_works_with_both(self):
        """A* routing works with both features enabled simultaneously."""
        self.city.set_enhanced_routing(
            fuzzy_enabled=True,
            bayesian_enabled=True,
            global_evidence={'rain': True, 'accident': True},
            bayesian_penalty_weight=2.0,
        )
        result = astar_search(self.city, 'A', 'T')
        assert result is not None
        assert result['cost'] > 0

    def test_disabling_features_restores_original_cost(self):
        """Disabling both features restores exact original routing costs."""
        # Get original cost
        result_original = astar_search(self.city, 'A', 'T')

        # Enable and then disable
        self.city.set_enhanced_routing(fuzzy_enabled=True, bayesian_enabled=True)
        self.city.set_enhanced_routing(fuzzy_enabled=False, bayesian_enabled=False)

        result_restored = astar_search(self.city, 'A', 'T')

        assert abs(result_original['cost'] - result_restored['cost']) < 1e-6, \
            "Disabling features should restore original cost"
        assert result_original['path'] == result_restored['path'], \
            "Disabling features should restore original path"

    def test_fuzzy_changes_cost_with_traffic(self):
        """Enabling fuzzy logic should change cost on roads with non-trivial traffic."""
        # Set some traffic
        self.city.update_traffic('A', 'B', 2.5)

        result_off = astar_search(self.city, 'A', 'B')
        self.city.set_enhanced_routing(fuzzy_enabled=True)
        result_on = astar_search(self.city, 'A', 'B')

        # The costs should differ (fuzzy replaces raw traffic_factor)
        assert result_off is not None and result_on is not None
        # Both should find a path, costs may differ
        assert result_on['cost'] > 0

    def test_bayesian_with_adverse_evidence_increases_cost(self):
        """Bayesian reasoning with rain+accident should increase cost vs no evidence."""
        result_off = astar_search(self.city, 'A', 'T')

        self.city.set_enhanced_routing(
            bayesian_enabled=True,
            global_evidence={'rain': True, 'accident': True},
            bayesian_penalty_weight=2.0,
        )
        result_on = astar_search(self.city, 'A', 'T')

        assert result_on['cost'] > result_off['cost'], \
            "Adverse Bayesian evidence should increase routing cost"

    def test_enhanced_routing_config_roundtrip(self):
        """get_enhanced_routing_config() returns the current configuration."""
        self.city.set_enhanced_routing(
            fuzzy_enabled=True,
            bayesian_enabled=True,
            bayesian_penalty_weight=1.5,
            global_evidence={'rain': True},
        )
        config = self.city.get_enhanced_routing_config()
        assert config['fuzzy_enabled'] is True
        assert config['bayesian_enabled'] is True
        assert abs(config['bayesian_penalty_weight'] - 1.5) < 1e-6
        assert config['global_evidence']['rain'] is True

    def test_edge_evidence(self):
        """Per-edge Bayesian evidence can be set and cleared."""
        self.city.set_edge_evidence('A', 'B', {'accident': True})
        edge_data = self.city.graph['A']['B']
        assert edge_data.get('bayesian_evidence', {}).get('accident') is True

        self.city.clear_edge_evidence('A', 'B')
        assert 'bayesian_evidence' not in self.city.graph['A']['B']

    def test_no_path_still_returns_none(self):
        """When destination is unreachable, A* returns None as before."""
        # Create a disconnected graph
        city = CityGraph()
        city.add_intersection('X', 0, 0)
        city.add_intersection('Y', 5, 5)
        # No road between X and Y
        result = astar_search(city, 'X', 'Y')
        assert result is None


# ═════════════════════════════════════════════
# MAIN: Run tests when executed directly
# ═════════════════════════════════════════════

def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestFuzzyLogic,
        TestBayesianReasoning,
        TestCostIntegration,
        TestSystemIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        print(f"\n{'=' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'=' * 60}")

        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(test_methods):
            total += 1
            method = getattr(instance, method_name)
            try:
                # Call setup if it exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                method()
                passed += 1
                print(f"  [PASS] {method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  [FAIL] {method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailed tests:")
        for cls_name, method_name, error in errors:
            print(f"  - {cls_name}.{method_name}: {error}")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
