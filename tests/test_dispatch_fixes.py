"""
Regression tests for the dispatch fixes:
  1. A* and Dijkstra are optimal (checked against NetworkX on identical weights)
  2. CSP picks hospitals by road cost, falls back when one is unreachable
  3. C4 (capacity limit) is independent of C1 (beds)
  4. Dispatch state (ambulance / hospital beds) is updated when committed
  5. Dashboard shows the A* vs Dijkstra comparison and closes its figures

Run:  python tests/test_dispatch_fixes.py     or     python -m pytest tests -v
"""
import os, sys, random
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.city_data import (create_city_graph, create_hospitals,
                            create_ambulances, randomize_traffic)
from algorithms.astar import astar_search, dijkstra_search, compare_search_results
from algorithms.csp import hospital_csp
from models.hospital import Hospital


# ─── helpers ───
def _truth(city, s, g):
    G = nx.Graph()
    for u in city.graph.nodes:
        for v, w in city.get_neighbors(u):
            G.add_edge(u, v, weight=w)
    return nx.dijkstra_path_length(G, s, g)


def _feasible(h, icu):
    return (h.available_beds > 0 and (not icu or h.icu_available)
            and h.doctors_available > 0 and (h.total_beds - h.available_beds) < h.capacity_limit)


# ─── 1. optimality ───
def test_astar_and_dijkstra_are_optimal_all_feature_combinations():
    rnd = random.Random(1)
    combos = [(False, False), (True, False), (False, True), (True, True)]
    for _ in range(120):
        city = create_city_graph()
        randomize_traffic(city)
        for fe, be in combos:
            city.set_enhanced_routing(fe, be, rnd.choice([0, 1, 2, 3]),
                                      {'rain': rnd.random() < .5, 'accident': rnd.random() < .5})
            for _ in range(4):
                s, g = rnd.sample(city.get_nodes(), 2)
                t = _truth(city, s, g)
                a, d = astar_search(city, s, g), dijkstra_search(city, s, g)
                assert abs(a['cost'] - t) < 1e-9, (fe, be, s, g, a['cost'], t)
                assert abs(d['cost'] - t) < 1e-9, (fe, be, s, g, d['cost'], t)
                pc = sum(dict(city.get_neighbors(u))[v] for u, v in zip(a['path'], a['path'][1:]))
                assert abs(pc - a['cost']) < 1e-9


def test_compare_search_results_helper():
    city = create_city_graph()
    a, d = astar_search(city, 'A', 'T'), dijkstra_search(city, 'A', 'T')
    cmp = compare_search_results(a, d)
    assert cmp['costs_match'] and cmp['nodes_saved'] >= 0
    assert compare_search_results(None, d) is None


# ─── 2. CSP road-cost selection and reachability ───
def test_csp_selects_by_road_cost_not_straight_line():
    rnd = random.Random(2)
    for _ in range(100):
        city = create_city_graph()
        randomize_traffic(city)
        em = rnd.choice(city.get_nodes())
        hs = create_hospitals()
        r = hospital_csp(hs, False, em, city)
        ok = [h for h in hs if _feasible(h, False)]
        best = min(astar_search(city, em, h.node)['cost'] for h in ok)
        assert abs(r['route_costs'][r['assigned_hospital'].name] - best) < 1e-9


def test_csp_never_violates_constraints():
    rnd = random.Random(7)
    city = create_city_graph()
    for _ in range(500):
        hs = []
        for i, n in enumerate(rnd.sample(city.get_nodes(), 5)):
            tb = rnd.randint(5, 60)
            hs.append(Hospital(f"H{i}", n, tb, rnd.randint(0, tb), rnd.random() < .5,
                               rnd.randint(0, 6), rnd.choice([tb, rnd.randint(1, tb)])))
        icu, em = rnd.random() < .5, rnd.choice(city.get_nodes())
        r = hospital_csp(hs, icu, em, city)
        ok = [h for h in hs if _feasible(h, icu)]
        if not ok:
            assert r['assigned_hospital'] is None and r['failure_reason']
        else:
            assert _feasible(r['assigned_hospital'], icu)


def _isolate(city, node):
    for n in list(city.graph.neighbors(node)):
        city.graph.remove_edge(node, n)


def test_csp_skips_unreachable_hospital_and_falls_back():
    city = create_city_graph()
    hs = create_hospitals()
    _isolate(city, 'C')                      # City General (node C) cut off
    r = hospital_csp(hs, False, 'G', city)   # G is right next to C
    assert r['assigned_hospital'] is not None
    assert r['assigned_hospital'].name != "City General Hospital"
    assert "City General Hospital" in r['unreachable']


def test_csp_reports_when_only_feasible_hospital_is_unreachable():
    city = create_city_graph()
    hs = create_hospitals()
    for h in hs:
        if h.name != "National Trauma Center":
            h.available_beds = 0
    _isolate(city, 'P')
    r = hospital_csp(hs, True, 'A', city)
    assert r['assigned_hospital'] is None
    assert "National Trauma Center" in r['unreachable']
    assert 'reachable' in r['failure_reason'].lower()


def test_csp_edge_cases():
    city = create_city_graph()
    assert hospital_csp([], False, 'A', city)['assigned_hospital'] is None
    hs = create_hospitals()
    for h in hs:
        h.icu_available = False
    r = hospital_csp(hs, True, 'Q', city)
    assert r['assigned_hospital'] is None and 'ICU' in r['failure_reason']
    hs = create_hospitals()
    for h in hs:
        h.doctors_available = 0
    assert hospital_csp(hs, False, 'Q', city)['assigned_hospital'] is None


def test_csp_traffic_changes_hospital_choice():
    """Straight-line nearest is St. Mary's (H) from L; heavy traffic on H-L makes
    the road-cheapest hospital National Trauma Center (P) instead."""
    city = create_city_graph()
    base = hospital_csp(create_hospitals(), False, 'L', city)['assigned_hospital'].name
    assert base == "St. Mary's Medical Center"
    city.update_traffic('H', 'L', 3.0)
    after = hospital_csp(create_hospitals(), False, 'L', city)['assigned_hospital'].name
    assert after == "National Trauma Center"


# ─── 3. C4 independent of C1 ───
def test_capacity_limit_is_independent_of_beds():
    hs = create_hospitals()
    assert any(h.capacity_limit < h.total_beds for h in hs)
    westside = next(h for h in hs if h.name == "Westside Clinic")
    westside.available_beds = 2              # occupied 18 >= limit 18, but beds > 0
    r = hospital_csp(hs, False, 'E', create_city_graph())
    assert "Westside Clinic" in r['domain_reductions'][4]['removed']   # removed by C4
    assert "Westside Clinic" not in r['domain_reductions'][1]['removed']  # NOT by C1
    assert not westside.has_capacity() and westside.admit_patient() is False


# ─── 4 & 5. Dashboard (Streamlit stubbed so this runs headless) ───
def _load_app():
    st = MagicMock(name='streamlit')

    class SS(dict):
        __getattr__ = dict.get
        def __setattr__(s, k, v): s[k] = v
    st.session_state = SS()
    log = []
    for k in ['error', 'success', 'info', 'warning', 'subheader', 'header',
              'markdown', 'text', 'title', 'caption']:
        setattr(st, k, (lambda kind: lambda m=None, *a, **kw: log.append((kind, str(m))))(k))
    st.metric = lambda l, v, *a, **k: log.append(('metric', f"{l}={v}"))

    class Ctx(MagicMock):
        def __enter__(s): return s
        def __exit__(s, *a): return False
        def metric(s, l, v, *a, **k): log.append(('metric', f"{l}={v}"))
    st.columns = lambda n, *a, **k: [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.expander = lambda *a, **k: Ctx()
    st.progress = lambda *a, **k: MagicMock()
    st.pyplot = lambda fig, *a, **k: log.append(('pyplot', ''))
    sys.modules['streamlit'] = st
    sys.modules.pop('app', None)
    import app
    return app, log


def _fresh(app, mutate=None):
    ss = app.st.session_state
    ss.city_graph = create_city_graph()
    ss.hospitals = create_hospitals()
    ss.ambulances = create_ambulances()
    if mutate:
        mutate(ss)
    return ss


def _msgs(log, kind):
    return [m for k, m in log if k == kind]


def test_dashboard_scenarios_run_and_show_dijkstra_comparison():
    app, log = _load_app()
    for sc in app.get_emergency_scenarios():
        _fresh(app); log.clear()
        app.run_dispatch(sc['location'], sc['severity'], sc['icu_required'], commit=False)
        assert not _msgs(log, 'error'), _msgs(log, 'error')
        assert any('agree on the optimal cost' in m for m in _msgs(log, 'success'))
        assert any('Dijkstra cost' in m for m in _msgs(log, 'metric'))


def test_dashboard_dispatch_commits_state_and_figures_are_closed():
    app, log = _load_app()
    ss = _fresh(app); log.clear()
    plt.close('all')
    app.run_dispatch('Q', 'Critical', True, commit=True)
    assert sum(1 for a in ss.ambulances if not a.is_available) == 2   # AMB-04 was already busy
    assert plt.get_fignums() == []                                     # no leaked figures
    ntc = next(h for h in ss.hospitals if h.name == "National Trauma Center")
    assert ntc.available_beds == 19 and ntc.doctors_available == 9


def test_dashboard_dry_run_leaves_state_untouched():
    app, log = _load_app()
    ss = _fresh(app)
    app.run_dispatch('Q', 'Critical', True, commit=False)
    assert all(a.is_available for a in ss.ambulances[:3])
    assert next(h for h in ss.hospitals if h.name == "National Trauma Center").available_beds == 20


def test_dashboard_edge_cases_report_errors():
    app, log = _load_app()
    def no_amb(ss):
        for a in ss.ambulances: a.is_available = False
    _fresh(app, no_amb); log.clear()
    app.run_dispatch('Q', 'High', False)
    assert any('No ambulances available' in m for m in _msgs(log, 'error'))

    def island(ss): ss.city_graph.add_intersection('Z', 20, 20)
    _fresh(app, island); log.clear()
    app.run_dispatch('Z', 'High', False)
    assert any('No path found' in m for m in _msgs(log, 'error'))

    def no_icu(ss):
        for h in ss.hospitals: h.icu_available = False
    _fresh(app, no_icu); log.clear()
    app.run_dispatch('Q', 'Critical', True)
    assert any('No suitable hospital' in m for m in _msgs(log, 'error'))

    def only_p_unreachable(ss):
        for h in ss.hospitals:
            if h.name != "National Trauma Center": h.available_beds = 0
        _isolate(ss.city_graph, 'P')
    ss = _fresh(app, only_p_unreachable); log.clear()
    app.run_dispatch('A', 'Critical', True)
    errs = _msgs(log, 'error')
    assert any('No suitable hospital' in m and 'reachable' in m.lower() for m in errs), errs
    assert all(a.is_available for a in ss.ambulances[:3])              # nothing committed


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"  [PASS] {n}")
        except Exception as e:
            failed += 1; print(f"  [FAIL] {n}: {type(e).__name__}: {e}")
    print(f"\nRESULTS: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
