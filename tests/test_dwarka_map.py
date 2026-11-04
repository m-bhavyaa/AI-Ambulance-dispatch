"""
Tests for the Dwarka real-map mode.

OpenStreetMap is not queried here. A small synthetic Overpass-format response
(a street grid placed at Dwarka's coordinates) is used so the OSM builder,
graph loader, routing, hospital CSP and map output are verified offline.
The fixture is test-only data, not real Dwarka roads.

Run:  python tests/test_dwarka_map.py     or     python -m pytest tests -v
"""
import json, math, os, random, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from data.osm_builder import (build_graph_from_overpass, build_query,
                              haversine_km, DWARKA_BBOX)
from data.dwarka_data import (create_dwarka_city_graph, create_dwarka_hospitals,
                              create_dwarka_ambulances, create_dwarka_scenarios,
                              load_dwarka_world, DwarkaDataMissing)
from data.city_data import randomize_traffic
from algorithms.astar import astar_search, dijkstra_search
from algorithms.csp import hospital_csp
from visualization.map_viz import route_geometry, build_map_html

ROWS, COLS = 4, 5
LAT0, LON0 = 28.580, 77.030
DLAT, DLON = 0.006, 0.008


def _gid(i, j):
    return 1000 + i * 10 + j


def _pt(i, j):
    return LAT0 + i * DLAT, LON0 + j * DLON


def _way(way_id, nodes_pts, tags):
    return {"type": "way", "id": way_id,
            "nodes": [n for n, _ in nodes_pts],
            "geometry": [{"lat": p[0], "lon": p[1]} for _, p in nodes_pts],
            "tags": tags}


def synthetic_payload():
    """4x5 street grid with curved (non-straight) roads + hospitals."""
    elements, wid = [], 1
    # Horizontal roads: each hop has a mid node bent sideways, so the road is
    # longer than the straight line (like a real curved road).
    for i in range(ROWS):
        pts = []
        for j in range(COLS):
            if j:
                a, b = _pt(i, j - 1), _pt(i, j)
                pts.append((50000 + _gid(i, j), ((a[0] + b[0]) / 2 + 0.0015, (a[1] + b[1]) / 2)))
            pts.append((_gid(i, j), _pt(i, j)))
        elements.append(_way(wid, pts, {"highway": "secondary", "name": f"Test Marg {i}"}))
        wid += 1
    for j in range(COLS):
        pts = []
        for i in range(ROWS):
            if i:
                a, b = _pt(i - 1, j), _pt(i, j)
                pts.append((60000 + _gid(i, j), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.0012)))
            pts.append((_gid(i, j), _pt(i, j)))
        elements.append(_way(wid, pts, {"highway": "primary", "name": f"Test Road {j}"}))
        wid += 1
    # A disconnected stub that must be dropped (not in the largest component)
    elements.append(_way(wid, [(9001, (28.6, 77.07)), (9002, (28.601, 77.071))],
                         {"highway": "tertiary", "name": "Isolated Lane"}))
    # Hospitals: nodes, a way with center, a duplicate, an unnamed, a far-away one
    for k, (i, j, name) in enumerate([(0, 1, "Alpha Hospital"), (2, 3, "Beta Medical"),
                                      (3, 0, "Gamma Care")]):
        lat, lon = _pt(i, j)
        elements.append({"type": "node", "id": 7000 + k, "lat": lat + 0.0003, "lon": lon + 0.0003,
                         "tags": {"amenity": "hospital", "name": name,
                                  **({"emergency": "yes"} if k == 0 else {}),
                                  **({"beds": "150"} if k == 1 else {})}})
    elements.append({"type": "way", "id": 7100, "center": {"lat": 28.5905, "lon": 77.044},
                     "tags": {"amenity": "hospital", "name": "Delta Centre"}})
    elements.append({"type": "node", "id": 7200, "lat": 28.58, "lon": 77.031,
                     "tags": {"amenity": "hospital", "name": "alpha hospital"}})   # dup name
    elements.append({"type": "node", "id": 7201, "lat": 28.58, "lon": 77.031,
                     "tags": {"amenity": "hospital"}})                              # unnamed
    elements.append({"type": "node", "id": 7202, "lat": 28.70, "lon": 77.20,
                     "tags": {"amenity": "hospital", "name": "Far Away Hospital"}})  # off-network
    return {"elements": elements}


def _graph_data():
    return build_graph_from_overpass(synthetic_payload())


def _write_world(tmpdir):
    path = os.path.join(tmpdir, "dwarka_graph.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_graph_data(), fh)
    return path


# ─── builder ───
def test_query_targets_dwarka_roads_and_hospitals():
    q = build_query()
    assert "amenity" in q and "hospital" in q and "out geom" in q
    s, w, n, e = DWARKA_BBOX
    assert f"{s},{w},{n},{e}" in q
    assert "residential" not in build_query(detail="major")
    assert "residential" in build_query(detail="all")


def test_builder_splits_at_junctions_and_drops_islands():
    d = _graph_data()
    # 4x5 grid -> 20 junction nodes, mid (curve) nodes are NOT graph nodes
    assert len(d["nodes"]) == ROWS * COLS, len(d["nodes"])
    assert not any(n in d["nodes"] for n in ("9001", "9002"))      # island dropped
    assert len(d["edges"]) == ROWS * (COLS - 1) + COLS * (ROWS - 1)
    for e in d["edges"]:
        nu, nv = d["nodes"][e["u"]], d["nodes"][e["v"]]
        straight = haversine_km(nu["lat"], nu["lon"], nv["lat"], nv["lon"])
        assert e["length_km"] > straight, "curved road must be longer than straight line"
        assert len(e["geom"]) == 3                                   # start, bend, end


def test_builder_labels_and_junction_flags():
    d = _graph_data()
    inner = d["nodes"][str(_gid(1, 1))]
    assert inner["degree"] == 4 and inner["is_junction"]
    assert "Test Marg 1" in inner["label"] and "Test Road 1" in inner["label"]
    corner = d["nodes"][str(_gid(0, 0))]
    assert corner["degree"] == 2 and not corner["is_junction"]


def test_builder_hospitals_snapped_and_cleaned():
    d = _graph_data()
    names = [h["name"] for h in d["hospitals"]]
    assert names == sorted(["Alpha Hospital", "Beta Medical", "Gamma Care", "Delta Centre"]), names
    by = {h["name"]: h for h in d["hospitals"]}
    assert by["Alpha Hospital"]["node"] == str(_gid(0, 1)) and by["Alpha Hospital"]["emergency"]
    assert by["Beta Medical"]["beds"] == 150
    assert all(h["snap_m"] <= 600 for h in d["hospitals"])


def test_builder_rejects_empty_response():
    try:
        build_graph_from_overpass({"elements": []})
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty response")


# ─── loader / graph ───
def test_loader_missing_file_message():
    try:
        load_dwarka_world(os.path.join(tempfile.gettempdir(), "nope_dwarka.json"))
    except DwarkaDataMissing as exc:
        assert "fetch_dwarka_osm" in str(exc)
        return
    raise AssertionError("expected DwarkaDataMissing")


def test_astar_heuristic_is_admissible_on_real_lengths():
    city = create_dwarka_city_graph(_graph_data())
    for u, v, data in city.get_edges():
        assert data["base_distance"] >= city.euclidean_distance(u, v) - 1e-12
    assert city.is_real_map()


def test_astar_matches_dijkstra_and_networkx_with_random_traffic():
    city = create_dwarka_city_graph(_graph_data())
    random.seed(7)
    nodes = city.get_nodes()
    for _ in range(6):
        randomize_traffic(city)
        for s, g in [(random.choice(nodes), random.choice(nodes)) for _ in range(12)]:
            a, dj = astar_search(city, s, g), dijkstra_search(city, s, g)
            G = nx.Graph()
            for u in city.graph.nodes:
                for v, w in city.get_neighbors(u):
                    G.add_edge(u, v, weight=w)
            truth = nx.dijkstra_path_length(G, s, g)
            assert abs(a["cost"] - truth) < 1e-9 and abs(dj["cost"] - truth) < 1e-9


def test_world_objects_are_deterministic_and_consistent():
    d = _graph_data()
    city = create_dwarka_city_graph(d)
    h1, h2 = create_dwarka_hospitals(d), create_dwarka_hospitals(d)
    assert [(h.name, h.total_beds, h.available_beds, h.doctors_available) for h in h1] == \
           [(h.name, h.total_beds, h.available_beds, h.doctors_available) for h in h2]
    assert [h for h in h1 if h.name == "Beta Medical"][0].total_beds == 150   # OSM beds tag used
    assert all(h.node in city.graph for h in h1)
    ambs = create_dwarka_ambulances(city)
    assert len(ambs) == 4 and sum(a.is_available for a in ambs) == 3
    assert len({a.current_node for a in ambs}) == 4                          # spread out
    sc = create_dwarka_scenarios(city)
    assert len(sc) == 4 and all(s["location"] in city.graph for s in sc)


def test_hospital_csp_works_on_real_graph():
    d = _graph_data()
    city = create_dwarka_city_graph(d)
    hospitals = create_dwarka_hospitals(d)
    res = hospital_csp(hospitals, False, str(_gid(3, 4)), city)
    assert res["assigned_hospital"] is not None
    assert res["route_results"][res["assigned_hospital"].name]["path"][0] == str(_gid(3, 4))


def test_load_world_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        city, hospitals, ambulances, scenarios, meta = load_dwarka_world(_write_world(tmp))
    assert city.graph.number_of_nodes() == 20 and len(hospitals) == 4
    assert meta["hospital_count"] == 4 and city.label(str(_gid(1, 1)))


# ─── map / animation ───
def test_route_geometry_follows_real_road_shape_and_traffic():
    city = create_dwarka_city_graph(_graph_data())
    a, b = str(_gid(0, 0)), str(_gid(2, 2))
    path = astar_search(city, a, b)["path"]
    g = route_geometry(city, path, 60)
    start, end = city.graph.nodes[a], city.graph.nodes[b]
    assert abs(g["points"][0][0] - start["lat"]) < 1e-5 and abs(g["points"][-1][1] - end["lon"]) < 1e-5
    assert all(x <= y for x, y in zip(g["cum_min"], g["cum_min"][1:]))
    assert len(g["points"]) > len(path)                          # includes curve points
    expected = sum(city.graph[u][v]["base_distance"] * city.graph[u][v]["traffic_factor"]
                   for u, v in zip(path[:-1], path[1:])) / 60 * 60
    assert abs(g["eta_min"] - expected) < 0.01
    # congestion slows the ambulance down
    for u, v in zip(path[:-1], path[1:]):
        city.update_traffic(u, v, 2.0)
    assert route_geometry(city, path, 60)["eta_min"] > g["eta_min"] * 1.9


def test_route_geometry_direction_independent_of_edge_storage():
    city = create_dwarka_city_graph(_graph_data())
    a, b = str(_gid(0, 0)), str(_gid(0, 1))
    fwd, back = route_geometry(city, [a, b], 60), route_geometry(city, [b, a], 60)
    assert fwd["points"] == back["points"][::-1]


def test_route_geometry_none_for_demo_city():
    from data.city_data import create_city_graph
    assert route_geometry(create_city_graph(), ['A', 'B'], 60) is None


def test_map_html_contains_everything_and_is_script_safe():
    d = _graph_data()
    d["edges"][0]["name"] = "</script><script>alert(1)</script>"          # hostile road name
    city = create_dwarka_city_graph(d)
    hospitals, ambs = create_dwarka_hospitals(d), create_dwarka_ambulances(city)
    amb = ambs[0]
    emergency = str(_gid(3, 4))
    r1 = astar_search(city, amb.current_node, emergency)
    h = hospitals[0]
    r2 = astar_search(city, emergency, h.node)
    legs = [{"name": "to patient", "color": "#2980b9", "geometry": route_geometry(city, r1["path"], amb.speed)},
            {"name": "to hospital", "color": "#c0392b", "geometry": route_geometry(city, r2["path"], amb.speed)}]
    html = build_map_html(city, hospitals, ambs, emergency_node=emergency, assigned_hospital=h,
                          legs=legs, explored_edges=r1["explored_edges"],
                          dispatched_ambulance_id=amb.ambulance_id)
    assert "__DATA__" not in html and "__HEIGHT__" not in html
    assert html.count("</script>") == 2                    # only the template's own two tags
    for name in ("Alpha Hospital", "Beta Medical", "Gamma Care", "Delta Centre"):
        assert name in html
    payload = html.split("const D = ", 1)[1].split(";\nconst esc", 1)[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert len(data["legs"]) == 2 and len(data["hospitals"]) == 4
    assert data["emergency"]["lat"] == city.graph.nodes[emergency]["lat"]
    assert [a["id"] for a in data["ambulances"]] == [a.ambulance_id for a in ambs[1:]]
    assert sum(h["assigned"] for h in data["hospitals"]) == 1
    assert len(data["roads"]) == city.graph.number_of_edges()


def test_map_html_without_route_has_no_animation_data():
    d = _graph_data()
    city = create_dwarka_city_graph(d)
    html = build_map_html(city, create_dwarka_hospitals(d), create_dwarka_ambulances(city))
    payload = html.split("const D = ", 1)[1].split(";\nconst esc", 1)[0]
    assert json.loads(payload)["legs"] == []


# ─── app integration (Streamlit stubbed, like test_dispatch_fixes) ───
def _real_map_app(tmp):
    from tests.test_dispatch_fixes import _load_app
    app, log = _load_app()
    captured = []
    app.show_map = lambda html, height=620: captured.append(html)
    city, hospitals, ambulances, scenarios, meta = load_dwarka_world(_write_world(tmp))
    ss = app.st.session_state
    ss.city_graph, ss.hospitals, ss.ambulances = city, hospitals, ambulances
    ss.scenarios, ss.world_meta, ss.map_mode = scenarios, meta, app.MODE_DWARKA
    ss.dispatch_result = None
    return app, log, captured, ss


def _map_data(html):
    payload = html.split("const D = ", 1)[1].split(";\nconst esc", 1)[0]
    return json.loads(payload.replace("<\\/", "</"))


def test_app_dispatch_in_real_map_mode_animates_both_legs():
    with tempfile.TemporaryDirectory() as tmp:
        app, log, captured, ss = _real_map_app(tmp)
        emergency = str(_gid(3, 4))
        app.run_dispatch(emergency, "Critical", True, commit=True)
        errors = [m for k, m in log if k == "error"]
        assert not errors, errors
        assert len(captured) == 1, "exactly one live map per dispatch"
        data = _map_data(captured[0])
        assert len(data["legs"]) == 2 and data["emergency"] is not None
        assert sum(h["assigned"] for h in data["hospitals"]) == 1
        assert data["legs"][0]["points"][-1] == data["legs"][1]["points"][0]   # leg 1 ends where leg 2 starts
        assert any(k == "metric" and m.startswith("Total ETA=") for k, m in log)
        assert sum(not a.is_available for a in ss.ambulances) == 2           # busy one + dispatched
        # the node-diagram figures are skipped; only the CSP charts are drawn
        assert sum(1 for k, _ in log if k == "pyplot") == 4


def test_app_dispatch_dry_run_leaves_fleet_unchanged_on_real_map():
    with tempfile.TemporaryDirectory() as tmp:
        app, log, captured, ss = _real_map_app(tmp)
        before = [a.is_available for a in ss.ambulances]
        app.run_dispatch(str(_gid(2, 1)), "Low", False, commit=False)
        assert [a.is_available for a in ss.ambulances] == before and len(captured) == 1


def test_app_tabs_render_in_real_map_mode():
    with tempfile.TemporaryDirectory() as tmp:
        app, log, captured, ss = _real_map_app(tmp)
        st = app.st
        st.selectbox = lambda label, options, *a, **k: options[0]
        st.checkbox = lambda label, value=False, *a, **k: value
        st.button = lambda *a, **k: False
        st.slider = lambda label, lo, hi, value=None, *a, **k: value
        st.number_input = lambda label, lo, hi, value=None, *a, **k: value
        st.container = lambda *a, **k: __import__("unittest.mock").mock.MagicMock()
        st.tabs = lambda names: [__import__("unittest.mock").mock.MagicMock() for _ in names]
        app.emergency_dispatch_tab()
        assert len(captured) == 1 and _map_data(captured[0])["legs"] == []     # overview, no route
        app.hospital_management_tab()
        app.system_admin_tab()
        assert len(captured) == 2                                              # admin map


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
