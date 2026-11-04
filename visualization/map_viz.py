"""
Interactive Dwarka map (Leaflet + OpenStreetMap tiles) with an animated
ambulance. Rendered inside Streamlit with st.components.v1.html, so it needs
no extra Python dependency; the browser loads Leaflet and the map tiles.

Public API:
    route_geometry(city, path, speed_kmh) -> dict | None
    build_map_html(city, hospitals, ambulances, ...) -> str
    show_map(html, height)                            (Streamlit helper)
"""

import json

from data.osm_builder import haversine_km

ROAD_WEIGHT = {
    "motorway": 5, "trunk": 5, "primary": 4.5, "secondary": 4,
    "tertiary": 3, "unclassified": 2.5, "residential": 2,
}


def route_geometry(city, path, speed_kmh):
    """
    Turn a node path into a drawable polyline with a travel-time profile.

    Travel time on each road = length * traffic_factor / speed, so the
    ambulance visibly slows down in congested roads.

    Returns:
        dict(points=[[lat, lon], ...], cum_min=[minutes at each point],
             length_km, eta_min), or None when the path is empty / has no
        real-map geometry.
    """
    if not path or len(path) < 2 or not city.is_real_map():
        return None
    speed = max(float(speed_kmh), 1.0)
    points, cum_min = [], []
    clock = 0.0
    for a, b in zip(path[:-1], path[1:]):
        edge = city.graph[a][b]
        geom = [list(p) for p in edge["geom"]]
        if edge.get("geom_from") != a:
            geom.reverse()
        edge_minutes = edge["base_distance"] * edge["traffic_factor"] / speed * 60.0
        sub = [haversine_km(p[0], p[1], q[0], q[1]) for p, q in zip(geom[:-1], geom[1:])]
        total = sum(sub) or 1.0
        for i, p in enumerate(geom):
            if i == 0 and points:
                continue  # joint with the previous road
            if i > 0:
                clock += edge_minutes * sub[i - 1] / total
            points.append([round(p[0], 6), round(p[1], 6)])
            cum_min.append(round(clock, 4))
    length_km = sum(city.graph[a][b]["base_distance"] for a, b in zip(path[:-1], path[1:]))
    return {"points": points, "cum_min": cum_min,
            "length_km": round(length_km, 3), "eta_min": round(clock, 2)}


def _traffic_color(tf):
    if tf <= 1.5:
        return "#2ecc71"
    if tf <= 2.0:
        return "#f39c12"
    return "#e74c3c"


def build_map_html(city, hospitals, ambulances, emergency_node=None,
                   assigned_hospital=None, legs=None, explored_edges=None,
                   dispatched_ambulance_id=None, height=620):
    """
    Build the standalone HTML page for the map.

    Args:
        city: CityGraph in real-map mode.
        hospitals / ambulances: model objects.
        emergency_node: node id of the emergency, if any.
        assigned_hospital: Hospital chosen by the CSP, if any.
        legs: list of dicts {name, color, geometry} where geometry comes
            from route_geometry(); animated in order.
        explored_edges: iterable of (u, v) A* explored edges (optional layer).
        dispatched_ambulance_id: ambulance that is animated (its static marker
            is hidden so it is not drawn twice).
        height: pixel height of the map.
    """
    G = city.graph

    roads = []
    for u, v, d in G.edges(data=True):
        roads.append({
            "g": [[round(p[0], 5), round(p[1], 5)] for p in d["geom"]],
            "c": _traffic_color(d["traffic_factor"]),
            "w": ROAD_WEIGHT.get(d.get("highway", "").replace("_link", ""), 2.5),
            "n": d.get("name", ""),
            "t": d["traffic_factor"],
            "k": round(d["base_distance"], 3),
        })

    junctions = [[round(d["lat"], 5), round(d["lon"], 5), d["label"]]
                 for _, d in G.nodes(data=True) if d.get("is_junction")]

    hosp = []
    for h in hospitals:
        n = G.nodes[h.node]
        hosp.append({
            "name": h.name, "lat": n["lat"], "lon": n["lon"],
            "beds": f"{h.available_beds}/{h.total_beds}",
            "doctors": h.doctors_available,
            "icu": bool(h.icu_available),
            "ok": bool(h.has_capacity()),
            "assigned": bool(assigned_hospital and h.name == assigned_hospital.name),
        })

    ambs = []
    for a in ambulances:
        if a.ambulance_id == dispatched_ambulance_id:
            continue
        n = G.nodes[a.current_node]
        ambs.append({"id": a.ambulance_id, "lat": n["lat"], "lon": n["lon"],
                     "free": bool(a.is_available), "speed": a.speed})

    emergency = None
    if emergency_node is not None:
        n = G.nodes[emergency_node]
        emergency = {"lat": n["lat"], "lon": n["lon"], "label": n["label"]}

    explored = []
    if explored_edges:
        seen = set()
        for u, v in explored_edges:
            key = (u, v) if u < v else (v, u)
            if key in seen or not G.has_edge(u, v):
                continue
            seen.add(key)
            explored.append([[round(p[0], 5), round(p[1], 5)] for p in G[u][v]["geom"]])

    leg_payload = []
    for leg in legs or []:
        g = leg.get("geometry")
        if g:
            leg_payload.append({"name": leg["name"], "color": leg["color"],
                                "points": g["points"], "cum": g["cum_min"],
                                "total": g["eta_min"], "km": g["length_km"]})

    # Dispatched ambulance drawn at the start of leg 1
    moving = None
    if leg_payload:
        moving = {"id": dispatched_ambulance_id or "AMB"}

    data = {"roads": roads, "junctions": junctions, "hospitals": hosp,
            "ambulances": ambs, "emergency": emergency, "explored": explored,
            "legs": leg_payload, "moving": moving}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    return (_TEMPLATE
            .replace("__HEIGHT__", str(int(height)))
            .replace("__DATA__", payload))


def show_map(html, height=620):
    """Render the map HTML in Streamlit."""
    import streamlit.components.v1 as components
    components.html(html, height=int(height) + 6, scrolling=False)


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin:0; padding:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
  #map { height: __HEIGHT__px; width: 100%; border-radius: 10px; }
  .pin { font-size: 22px; line-height: 26px; text-align:center; filter: drop-shadow(0 1px 2px rgba(0,0,0,.45)); }
  .hosp { background:#fff; border:2px solid #c0392b; border-radius:50%; width:28px; height:28px;
          text-align:center; line-height:24px; font-size:15px; box-shadow:0 1px 4px rgba(0,0,0,.4); }
  .hosp.assigned { border-color:#27ae60; border-width:3px; width:34px; height:34px; line-height:30px;
                   box-shadow:0 0 0 4px rgba(39,174,96,.35); }
  .hosp.full { border-color:#7f8c8d; opacity:.65; }
  .sos { animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%{transform:scale(1)} 50%{transform:scale(1.25)} 100%{transform:scale(1)} }
  .amb-moving { font-size: 30px; filter: drop-shadow(0 2px 3px rgba(0,0,0,.6)); }
  .amb-idle { opacity:.85; }
  .amb-busy { opacity:.35; filter: grayscale(1); }
  .panel { background:rgba(255,255,255,.95); padding:8px 10px; border-radius:8px;
           box-shadow:0 1px 6px rgba(0,0,0,.35); font-size:12px; line-height:1.45; }
  .panel button, .panel select { font-size:12px; margin-right:4px; cursor:pointer; }
  .legend i { display:inline-block; width:14px; height:4px; margin-right:6px; vertical-align:middle; }
</style></head>
<body><div id="map"></div>
<script>
const D = __DATA__;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const map = L.map('map', { preferCanvas: true, zoomControl: true });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);
const canvas = L.canvas({ padding: 0.5 });

// ── roads coloured by traffic ──
const roadLayer = L.layerGroup().addTo(map);
D.roads.forEach(r => {
  const line = L.polyline(r.g, { color: r.c, weight: r.w, opacity: .8, renderer: canvas });
  line.bindTooltip(esc(r.n || 'Unnamed road') + '<br>' + r.k + ' km, traffic ' + r.t.toFixed(1) + '×', { sticky: true });
  line.addTo(roadLayer);
});

// ── junction dots (only when zoomed in) ──
const junctionLayer = L.layerGroup();
D.junctions.forEach(j => {
  L.circleMarker([j[0], j[1]], { radius: 3, color: '#2c3e50', weight: 1, fillColor: '#fff', fillOpacity: 1, renderer: canvas })
    .bindTooltip(esc(j[2])).addTo(junctionLayer);
});
function syncJunctions() {
  if (map.getZoom() >= 15) { if (!map.hasLayer(junctionLayer)) junctionLayer.addTo(map); }
  else if (map.hasLayer(junctionLayer)) map.removeLayer(junctionLayer);
}
map.on('zoomend', syncJunctions);

// ── A* explored edges (toggle) ──
const exploredLayer = L.layerGroup();
D.explored.forEach(g => L.polyline(g, { color: '#8e44ad', weight: 2, opacity: .7, dashArray: '4 4', renderer: canvas }).addTo(exploredLayer));

// ── hospitals ──
const hospLayer = L.layerGroup().addTo(map);
D.hospitals.forEach(h => {
  const cls = 'hosp' + (h.assigned ? ' assigned' : '') + (h.ok ? '' : ' full');
  const size = h.assigned ? 34 : 28;
  const icon = L.divIcon({ className: '', html: '<div class="' + cls + '">🏥</div>', iconSize: [size, size], iconAnchor: [size/2, size/2] });
  L.marker([h.lat, h.lon], { icon, zIndexOffset: h.assigned ? 900 : 500 })
    .bindPopup('<b>' + esc(h.name) + '</b><br>Beds free: ' + esc(h.beds) + '<br>Doctors: ' + h.doctors +
               '<br>ICU: ' + (h.icu ? 'yes' : 'no') + (h.ok ? '' : '<br><i>Not accepting patients</i>') +
               '<br><small>Capacity figures are simulated.</small>')
    .bindTooltip(esc(h.name) + (h.assigned ? ' (assigned)' : ''))
    .addTo(hospLayer);
});

// ── idle ambulances ──
const ambLayer = L.layerGroup().addTo(map);
D.ambulances.forEach(a => {
  const icon = L.divIcon({ className: '', html: '<div class="pin ' + (a.free ? 'amb-idle' : 'amb-busy') + '">🚑</div>', iconSize: [26, 26], iconAnchor: [13, 13] });
  L.marker([a.lat, a.lon], { icon, zIndexOffset: 600 })
    .bindTooltip(esc(a.id) + (a.free ? ' (available)' : ' (busy)')).addTo(ambLayer);
});

// ── emergency ──
if (D.emergency) {
  const icon = L.divIcon({ className: '', html: '<div class="pin sos">🆘</div>', iconSize: [28, 28], iconAnchor: [14, 14] });
  L.marker([D.emergency.lat, D.emergency.lon], { icon, zIndexOffset: 1000 })
    .bindTooltip('Emergency: ' + esc(D.emergency.label)).addTo(map);
}

// ── routes ──
const routeLayer = L.layerGroup().addTo(map);
D.legs.forEach(leg => {
  L.polyline(leg.points, { color: leg.color, weight: 6, opacity: .35 }).addTo(routeLayer);
});
const trail = L.polyline([], { weight: 6, opacity: 1 }).addTo(map);

// ── layer control + legend ──
L.control.layers(null, { 'Roads (traffic)': roadLayer, 'A* explored edges': exploredLayer,
                         'Hospitals': hospLayer, 'Ambulances': ambLayer, 'Routes': routeLayer },
                 { collapsed: true }).addTo(map);
const legend = L.control({ position: 'bottomleft' });
legend.onAdd = () => {
  const d = L.DomUtil.create('div', 'panel legend');
  d.innerHTML = '<i style="background:#2ecc71"></i>clear &nbsp;<i style="background:#f39c12"></i>moderate &nbsp;<i style="background:#e74c3c"></i>heavy traffic';
  return d;
};
legend.addTo(map);

// ── fit view ──
const bounds = [];
if (D.legs.length) D.legs.forEach(l => l.points.forEach(p => bounds.push(p)));
else D.hospitals.forEach(h => bounds.push([h.lat, h.lon]));
if (!bounds.length) D.roads.forEach(r => r.g.forEach(p => bounds.push(p)));
if (bounds.length) map.fitBounds(bounds, { padding: [40, 40] });
syncJunctions();

// ── ambulance animation ──
if (D.legs.length) {
  const ambIcon = L.divIcon({ className: '', html: '<div class="pin amb-moving">🚑</div>', iconSize: [34, 34], iconAnchor: [17, 17] });
  const first = D.legs[0].points[0];
  const marker = L.marker(first, { icon: ambIcon, zIndexOffset: 2000 }).addTo(map);

  const panel = L.control({ position: 'topright' });
  let statusEl, playBtn, speedSel;
  panel.onAdd = () => {
    const d = L.DomUtil.create('div', 'panel');
    d.innerHTML = '<div id="status"></div><div style="margin-top:4px">' +
      '<button id="play">⏸</button><button id="restart">⟲ Replay</button>' +
      '<select id="speed"><option value="5">5×</option><option value="15">15×</option>' +
      '<option value="30" selected>30×</option><option value="60">60×</option><option value="120">120×</option></select></div>';
    L.DomEvent.disableClickPropagation(d);
    return d;
  };
  panel.addTo(map);
  statusEl = document.getElementById('status');
  playBtn = document.getElementById('play');
  speedSel = document.getElementById('speed');

  const totalMin = D.legs.reduce((s, l) => s + l.total, 0);
  let legIdx = 0, t = 0, elapsedBefore = 0, playing = true, holdUntil = 0, last = null;

  function pointAt(leg, tm) {
    const c = leg.cum;
    if (tm <= 0) return { p: leg.points[0], i: 0 };
    if (tm >= leg.total) return { p: leg.points[leg.points.length - 1], i: leg.points.length - 1 };
    let lo = 0, hi = c.length - 1;
    while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (c[mid] <= tm) lo = mid; else hi = mid; }
    const span = c[hi] - c[lo] || 1e-9, f = (tm - c[lo]) / span;
    const a = leg.points[lo], b = leg.points[hi];
    return { p: [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f], i: lo };
  }
  function fmt(m) { const s = Math.max(0, Math.round(m * 60)); return Math.floor(s / 60) + ' min ' + String(s % 60).padStart(2, '0') + ' s'; }
  function render(done) {
    const leg = D.legs[legIdx];
    const { p, i } = pointAt(leg, t);
    marker.setLatLng(p);
    trail.setStyle({ color: leg.color });
    trail.setLatLngs(leg.points.slice(0, i + 1).concat([p]));
    const elapsed = elapsedBefore + t;
    statusEl.innerHTML = '<b>' + esc(leg.name) + '</b> (' + leg.km + ' km)<br>' +
      (done ? '✅ Arrived. Total ' + fmt(totalMin) : 'Elapsed ' + fmt(elapsed) + ' · ETA left ' + fmt(totalMin - elapsed));
  }
  function restart() { legIdx = 0; t = 0; elapsedBefore = 0; holdUntil = 0; last = null; playing = true; playBtn.textContent = '⏸'; render(false); requestAnimationFrame(tick); }
  function tick(now) {
    if (!playing) { last = null; return; }
    if (last === null) last = now;
    const dt = Math.min((now - last) / 1000, 0.25); last = now;
    if (now < holdUntil) { requestAnimationFrame(tick); return; }
    t += dt * parseFloat(speedSel.value) / 60;
    const leg = D.legs[legIdx];
    if (t >= leg.total) {
      t = leg.total; render(false);
      if (legIdx < D.legs.length - 1) {
        elapsedBefore += leg.total; legIdx += 1; t = 0; holdUntil = now + 1500;  // pause at the patient
        render(false);
      } else { render(true); playing = false; playBtn.textContent = '▶'; return; }
    } else { render(false); }
    requestAnimationFrame(tick);
  }
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '⏸' : '▶'; if (playing) { last = null; requestAnimationFrame(tick); } };
  document.getElementById('restart').onclick = restart;
  restart();
}
</script></body></html>
"""
