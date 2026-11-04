"""
AI-Based Emergency Ambulance Dispatch & Hospital Optimization System
Main Streamlit Application

CS 2005 - Artificial Intelligence Project
Uses A* Search for ambulance routing and CSP for hospital selection.
"""

import streamlit as st
import matplotlib.pyplot as plt
import random
import copy

from data.city_data import (
    create_city_graph, create_hospitals, create_ambulances,
    randomize_traffic, get_emergency_scenarios
)
from algorithms.astar import astar_search, dijkstra_search, compare_search_results
from algorithms.csp import hospital_csp
from visualization.graph_viz import (
    draw_city_graph, draw_astar_exploration, draw_comparison
)
from visualization.csp_viz import (
    draw_csp_domain_reduction, draw_csp_backtracking,
    draw_csp_constraints, draw_csp_assignment
)
from algorithms.fuzzy_logic import compute_fuzzy_congestion, congestion_score_to_factor
from algorithms.bayesian_reasoning import compute_bayesian_risk, risk_probability_to_factor
from algorithms.cost_integration import compute_effective_weight
from data.dwarka_data import load_dwarka_world, data_available, DwarkaDataMissing
from data.osm_builder import fetch_and_build
from visualization.map_viz import build_map_html, route_geometry, show_map

# ─── Page Configuration ───
st.set_page_config(
    page_title="AI Emergency Dispatch System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Map modes ───
MODE_DEMO = "🧪 Demo city (abstract nodes)"
MODE_DWARKA = "🗺️ Dwarka, Delhi (real map)"


def build_world(mode):
    """Return (city, hospitals, ambulances, scenarios, meta) for a map mode."""
    if mode == MODE_DWARKA:
        return load_dwarka_world()
    return (create_city_graph(), create_hospitals(), create_ambulances(),
            get_emergency_scenarios(), {})


def load_world(mode):
    """Load a map mode into the session, replacing the current world."""
    city, hospitals, ambulances, scenarios, meta = build_world(mode)
    st.session_state.city_graph = city
    st.session_state.hospitals = hospitals
    st.session_state.ambulances = ambulances
    st.session_state.scenarios = scenarios
    st.session_state.world_meta = meta
    st.session_state.map_mode = mode
    st.session_state.dispatch_result = None


# ─── Initialize Session State ───
if 'city_graph' not in st.session_state:
    load_world(MODE_DEMO)
if 'dispatch_result' not in st.session_state:
    st.session_state.dispatch_result = None


def show_fig(fig):
    """Render a matplotlib figure in Streamlit and close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


def fmt_path(city, path, max_hops=8):
    """Readable path: full for short demo paths, abbreviated for real maps."""
    if len(path) <= max_hops:
        return " → ".join(city.label(n) for n in path)
    return (f"{city.label(path[0])} → … ({len(path) - 2} junctions) → "
            f"{city.label(path[-1])}")


def dwarka_setup_screen():
    """Shown when Dwarka mode is selected but data/dwarka_graph.json is missing."""
    st.title("🗺️ Dwarka real-map mode")
    st.info(
        "The Dwarka road network and hospital locations come from OpenStreetMap "
        "and need to be downloaded once (about a minute, needs internet access)."
    )
    if st.button("⬇️ Download Dwarka map data from OpenStreetMap", type="primary"):
        with st.spinner("Downloading roads and hospitals for Dwarka..."):
            try:
                meta = fetch_and_build()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.markdown("You can also run this in a terminal and then reload:")
                st.code("pip install requests\npython scripts/fetch_dwarka_osm.py")
                return
        st.success(f"Saved {meta['node_count']} intersections, {meta['edge_count']} "
                   f"roads and {meta['hospital_count']} hospitals.")
        st.rerun()
    st.markdown("Or run it yourself:")
    st.code("pip install requests\npython scripts/fetch_dwarka_osm.py")


def show_search_comparison(label, astar_result, dijkstra_result, fig_args=None,
                           city=None):
    """Show the A* vs Dijkstra baseline comparison for one route leg."""
    cmp = compare_search_results(astar_result, dijkstra_result)
    if cmp is None:
        return None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label}: A* cost", f"{cmp['astar_cost']:.2f}")
    c2.metric(f"{label}: Dijkstra cost", f"{cmp['dijkstra_cost']:.2f}")
    c3.metric("Nodes expanded (A* / Dijkstra)",
              f"{cmp['astar_nodes_expanded']} / {cmp['dijkstra_nodes_expanded']}")
    c4.metric("Nodes saved by heuristic", cmp['nodes_saved'])
    if cmp['costs_match']:
        st.success(f"✅ {label}: A* and Dijkstra agree on the optimal cost.")
    else:
        st.error(f"⚠️ {label}: A* cost {cmp['astar_cost']:.4f} differs from "
                 f"Dijkstra cost {cmp['dijkstra_cost']:.4f}. The routing result "
                 f"is not trustworthy.")
    if fig_args is not None and city is not None:
        with st.expander(f"🔍 {label}: A* vs Dijkstra exploration map"):
            show_fig(draw_comparison(city, astar_result, dijkstra_result, **fig_args))
    return cmp


def main():
    """Main application entry point."""

    # ─── Sidebar Navigation ───
    st.sidebar.title("🚑 AI Dispatch System")
    st.sidebar.markdown("---")

    modes = [MODE_DEMO, MODE_DWARKA]
    chosen_mode = st.sidebar.radio("Map", modes,
                                   index=modes.index(st.session_state.map_mode))
    if chosen_mode != st.session_state.map_mode:
        if chosen_mode == MODE_DWARKA and not data_available():
            dwarka_setup_screen()
            return
        try:
            load_world(chosen_mode)
        except DwarkaDataMissing:
            dwarka_setup_screen()
            return
        st.rerun()
    if st.session_state.map_mode == MODE_DWARKA:
        st.sidebar.caption("Roads, intersections and hospital locations: "
                           "© OpenStreetMap contributors. Beds, doctors, traffic "
                           "and fleet are simulated.")
    st.sidebar.markdown("---")

    tab = st.sidebar.radio(
        "Navigation",
        [
            "🆘 Emergency Dispatch",
            "🏥 Hospital Management",
            "⚙️ System Administration",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        
        "AI-Based Emergency Ambulance\n"
        "Dispatch & Hospital Optimization"
    )

    # ─── Route to selected tab ───
    if tab == "🆘 Emergency Dispatch":
        emergency_dispatch_tab()
    elif tab == "🏥 Hospital Management":
        hospital_management_tab()
    elif tab == "⚙️ System Administration":
        system_admin_tab()


# ═══════════════════════════════════════════
# TAB 1: EMERGENCY DISPATCH
# ═══════════════════════════════════════════
def emergency_dispatch_tab():
    """Emergency Control Operator interface."""
    st.title("🆘 Emergency Dispatch")
    st.markdown("*Input emergency details to dispatch the nearest ambulance and select optimal hospital.*")

    city = st.session_state.city_graph
    hospitals = st.session_state.hospitals
    ambulances = st.session_state.ambulances

    real_map = city.is_real_map()

    # ─── Show City Map (filled below, once the emergency location is chosen) ───
    st.subheader("🗺️ Dwarka Map Overview" if real_map else "City Map Overview")
    overview_slot = st.container()

    st.markdown("---")

    # ─── Emergency Input Form ───
    st.subheader("📋 Emergency Request")
    col1, col2, col3 = st.columns(3)

    with col1:
        nodes = city.get_selectable_nodes()
        emergency_location = st.selectbox(
            "Emergency Location (Intersection)" if real_map else "Emergency Location (Node)",
            nodes, index=0 if real_map else 16, format_func=city.label)

    with col2:
        severity = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])

    with col3:
        icu_required = st.checkbox("ICU Required", value=(severity == "Critical"))

    # Quick scenario loader
    st.markdown("**Or load a scenario:**")
    scenarios = st.session_state.scenarios
    scenario_names = ["-- Select --"] + [s['name'] for s in scenarios]
    selected_scenario = st.selectbox("Pre-defined Scenarios", scenario_names)

    if selected_scenario != "-- Select --":
        sc = next(s for s in scenarios if s['name'] == selected_scenario)
        emergency_location = sc['location']
        severity = sc['severity']
        icu_required = sc['icu_required']
        st.info(f"Loaded: {sc['name']} | Location: {city.label(sc['location'])} | Severity: {sc['severity']} | ICU: {sc['icu_required']}")

    # Overview map (now that the emergency location is known)
    with overview_slot:
        if real_map:
            show_map(build_map_html(city, hospitals, ambulances,
                                    emergency_node=emergency_location), 560)
        else:
            show_fig(draw_city_graph(city, hospitals, ambulances,
                                     title="City Road Network"))

    commit = st.checkbox(
        "Commit dispatch (mark ambulance busy and admit the patient)",
        value=True,
        help="When ticked, the chosen ambulance becomes unavailable and the assigned "
             "hospital loses one bed and one doctor. Untick for a dry run. "
             "Use System Administration to release ambulances."
    )

    st.markdown("---")

    # ─── Dispatch Button ───
    if st.button("🚨 DISPATCH AMBULANCE", type="primary", use_container_width=True):
        run_dispatch(emergency_location, severity, icu_required, commit=commit)


def run_dispatch(emergency_location, severity, icu_required, commit=True):
    """
    Run the full dispatch process: A* ambulance selection, CSP hospital
    selection (road-cost based), A* route to hospital, Dijkstra baseline.

    If commit is True the chosen ambulance is marked busy and the assigned
    hospital admits the patient, so later dispatches see the updated state.
    """
    city = st.session_state.city_graph
    hospitals = st.session_state.hospitals
    ambulances = st.session_state.ambulances

    real_map = city.is_real_map()

    st.markdown("---")
    st.header("📡 Dispatch Results")
    map_slot = st.container()  # live map is filled in once both route legs are known

    # Show active AI-enhanced routing status if enabled
    if city._fuzzy_enabled or city._bayesian_enabled:
        badges = []
        if city._fuzzy_enabled:
            badges.append("🧠 **Fuzzy Logic (Congestion)**")
        if city._bayesian_enabled:
            bayesian_info = "🎲 **Bayesian Reasoning (Road Risk**"
            ev_list = []
            if city._global_evidence.get('rain'):
                ev_list.append("🌧️ Rain (simulated)")
            if city._global_evidence.get('accident'):
                ev_list.append("🚗 Accident (simulated)")
            if ev_list:
                bayesian_info += f": {', '.join(ev_list)}"
            bayesian_info += f", {city._bayesian_penalty_weight:.1f}× penalty)"
            badges.append(bayesian_info)
        st.info(f"AI-Enhanced Routing Active: {' | '.join(badges)}")

    # ─── Step 1: Select nearest available ambulance ───
    st.subheader("Step 1: Select Nearest Ambulance (A* Search)")

    available = [a for a in ambulances if a.is_available]
    if not available:
        st.error("No ambulances available! Release an ambulance in System Administration.")
        return

    best_ambulance = None
    best_result = None
    best_cost = float('inf')

    progress = st.progress(0)
    for i, amb in enumerate(available):
        result = astar_search(city, amb.current_node, emergency_location)
        if result and result['cost'] < best_cost:
            best_cost = result['cost']
            best_ambulance = amb
            best_result = result
        progress.progress((i + 1) / len(available))

    if not best_ambulance:
        st.error("No path found to emergency location! No available ambulance can "
                 "reach it by road.")
        return

    st.success(f"**Selected Ambulance:** {best_ambulance.ambulance_id} at "
               f"{'node ' if not real_map else ''}{city.label(best_ambulance.current_node)} "
               f"(cost: {best_cost:.2f})")

    # Show A* exploration visualization (real-map mode shows it on the live map)
    if not real_map:
        fig_astar = draw_astar_exploration(
            city, best_result, hospitals, ambulances,
            emergency_node=emergency_location,
            ambulance_node=best_ambulance.current_node
        )
        show_fig(fig_astar)

    # Show A* steps table
    with st.expander("📋 A* Search Steps Detail"):
        for step in best_result['steps']:
            icon = "🔍" if step['action'] == 'expand' else "🎯"
            st.text(f"{icon} Node {city.label(step['current_node'])}: g={step['g_score']:.2f}, "
                    f"h={step['h_score']:.2f}, f={step['f_score']:.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes Expanded", best_result['nodes_expanded'])
    col2.metric("Path Cost", f"{best_result['cost']:.2f}")
    col3.metric("Path", fmt_path(city, best_result['path']))

    # Dijkstra baseline for leg 1 (same effective weights) + ambulance-choice check
    dijkstra_leg1 = dijkstra_search(city, best_ambulance.current_node, emergency_location)
    st.markdown("**A* vs Dijkstra baseline (Ambulance → Emergency)**")
    show_search_comparison(
        "Leg 1", best_result, dijkstra_leg1,
        fig_args=None if real_map else dict(
            ambulance_node=best_ambulance.current_node,
            emergency_node=emergency_location, hospital_node=None),
        city=city)
    dijkstra_costs = {}
    for amb in available:
        d = dijkstra_search(city, amb.current_node, emergency_location)
        if d:
            dijkstra_costs[amb.ambulance_id] = d['cost']
    if dijkstra_costs:
        dijkstra_best = min(dijkstra_costs.values())
        if abs(dijkstra_best - best_cost) > 1e-9:
            st.error("⚠️ Dijkstra finds a cheaper ambulance route than A*: "
                     f"{dijkstra_best:.2f} vs {best_cost:.2f}.")

    st.markdown("---")

    # ─── Step 2: Select hospital using CSP ───
    st.subheader("Step 2: Select Hospital (CSP)")

    csp_result = hospital_csp(hospitals, icu_required, emergency_location, city)
    assigned = csp_result['assigned_hospital']
    hospital_path_result = None

    if assigned:
        st.success(f"**Assigned Hospital:** {assigned.name} at "
                   f"{'node ' if not real_map else ''}{city.label(assigned.node)} "
                   f"(road cost {csp_result['route_costs'][assigned.name]:.2f})")
        if csp_result['unreachable']:
            st.warning("Hospitals meeting medical constraints but unreachable by road "
                       f"were skipped: {', '.join(csp_result['unreachable'])}")

        # Route Emergency → Hospital (already computed by the CSP's C5 check)
        hospital_path_result = csp_result['route_results'].get(assigned.name)
        if hospital_path_result is None:
            hospital_path_result = astar_search(city, emergency_location, assigned.node)

        if hospital_path_result:
            st.markdown("**Route: Emergency → Hospital**")
            if not real_map:
                fig_hosp = draw_astar_exploration(
                    city, hospital_path_result, hospitals, ambulances,
                    emergency_node=emergency_location,
                    ambulance_node=best_ambulance.current_node
                )
                show_fig(fig_hosp)

            total_cost = best_result['cost'] + hospital_path_result['cost']
            st.info(f"**Total Response Time Cost:** {total_cost:.2f} "
                    f"(Ambulance→Emergency: {best_result['cost']:.2f} + "
                    f"Emergency→Hospital: {hospital_path_result['cost']:.2f})")

            dijkstra_leg2 = dijkstra_search(city, emergency_location, assigned.node)
            st.markdown("**A* vs Dijkstra baseline (Emergency → Hospital)**")
            show_search_comparison(
                "Leg 2", hospital_path_result, dijkstra_leg2,
                fig_args=None if real_map else dict(
                    ambulance_node=emergency_location,
                    emergency_node=emergency_location,
                    hospital_node=assigned.node),
                city=city)

            # Real map: animate the ambulance along both legs
            if real_map:
                leg1 = route_geometry(city, best_result['path'], best_ambulance.speed)
                leg2 = route_geometry(city, hospital_path_result['path'],
                                      best_ambulance.speed)
                with map_slot:
                    st.subheader("🗺️ Live Ambulance Map")
                    if leg1 and leg2:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("To patient",
                                  f"{leg1['eta_min']:.1f} min · {leg1['length_km']:.1f} km")
                        m2.metric("To hospital",
                                  f"{leg2['eta_min']:.1f} min · {leg2['length_km']:.1f} km")
                        m3.metric("Total ETA",
                                  f"{leg1['eta_min'] + leg2['eta_min']:.1f} min")
                        st.caption(
                            f"ETA = road length × traffic factor ÷ {best_ambulance.ambulance_id} "
                            f"speed ({best_ambulance.speed} km/h). The animation is sped up; "
                            "use the buttons on the map to pause, replay or change speed.")
                    show_map(build_map_html(
                        city, hospitals, ambulances,
                        emergency_node=emergency_location,
                        assigned_hospital=assigned,
                        legs=[
                            {"name": f"{best_ambulance.ambulance_id} → patient",
                             "color": "#2980b9", "geometry": leg1},
                            {"name": f"{best_ambulance.ambulance_id} → {assigned.name}",
                             "color": "#c0392b", "geometry": leg2},
                        ],
                        explored_edges=best_result['explored_edges'],
                        dispatched_ambulance_id=best_ambulance.ambulance_id,
                    ), 620)
        else:
            st.error(f"No road route from {emergency_location} to {assigned.name}.")
            assigned = None
    else:
        reason = csp_result.get('failure_reason') or "All constraints failed."
        st.error(f"No suitable hospital found! {reason}")
        if csp_result['unreachable']:
            st.warning("Unreachable by road: " + ", ".join(csp_result['unreachable']))

    # CSP Visualizations
    st.markdown("#### CSP Constraint Analysis")
    show_fig(draw_csp_constraints(csp_result))

    st.markdown("#### CSP Domain Reduction")
    show_fig(draw_csp_domain_reduction(csp_result))

    st.markdown("#### CSP Backtracking Steps")
    show_fig(draw_csp_backtracking(csp_result))

    st.markdown("#### CSP Final Assignment")
    show_fig(draw_csp_assignment(csp_result, hospitals, city))

    # ─── Commit dispatch to system state ───
    if commit and assigned and hospital_path_result:
        beds_before = assigned.available_beds
        if assigned.admit_patient():
            best_ambulance.dispatch()
            st.success(
                f"System updated: {best_ambulance.ambulance_id} is now busy; "
                f"{assigned.name} beds {beds_before} → {assigned.available_beds}, "
                f"doctors now {assigned.doctors_available}. "
                f"(Maps refresh on your next interaction.)")
        else:
            st.warning(f"Could not admit patient at {assigned.name}; state not updated.")
    elif commit:
        st.warning("Dispatch not committed: no complete route (ambulance → emergency "
                   "→ hospital) was found.")

    # Store result
    st.session_state.dispatch_result = {
        'ambulance': best_ambulance,
        'astar_result': best_result,
        'csp_result': csp_result,
    }


# ═══════════════════════════════════════════
# TAB 2: HOSPITAL MANAGEMENT
# ═══════════════════════════════════════════
def hospital_management_tab():
    """Hospital Administrator interface."""
    st.title("🏥 Hospital Management")
    st.markdown("*Update hospital capacity and monitor patient allocation.*")

    hospitals = st.session_state.hospitals

    # ─── Hospital Capacity Dashboard ───
    st.subheader("Hospital Capacity Dashboard")

    cols_per_row = 3
    for i in range(0, len(hospitals), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(hospitals):
                h = hospitals[i + j]
                with cols[j]:
                    with st.container(border=True):
                        occupancy = ((h.total_beds - h.available_beds) / h.total_beds) * 100
                        color = "🟢" if occupancy < 60 else ("🟡" if occupancy < 85 else "🔴")
                        st.markdown(f"#### {color} {h.name}")
                        st.metric("Available Beds", f"{h.available_beds}/{h.total_beds}")
                        st.metric("Doctors", h.doctors_available)
                        st.metric("ICU", "✅ Yes" if h.icu_available else "❌ No")
                        st.metric("Occupied / Capacity Limit",
                                  f"{h.occupied_beds}/{h.capacity_limit}")
                        if not h.has_capacity():
                            st.caption("⛔ Not accepting new patients "
                                       "(beds, doctors or capacity limit reached).")
                        st.progress(1.0 - (h.available_beds / h.total_beds), text=f"Occupancy: {occupancy:.1f}%")

    st.markdown("---")

    # ─── Update Hospital Capacity ───
    st.subheader("Update Hospital Capacity")

    hospital_names = [h.name for h in hospitals]
    selected = st.selectbox("Select Hospital", hospital_names)
    h_idx = hospital_names.index(selected)
    h = hospitals[h_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        new_beds = st.number_input("Available Beds", 0, h.total_beds, h.available_beds)
    with col2:
        new_doctors = st.number_input("Available Doctors", 0,
                                      max(20, h.doctors_available + 20),
                                      h.doctors_available)
    with col3:
        new_icu = st.checkbox("ICU Available", h.icu_available)

    if st.button("💾 Update Hospital", type="primary"):
        h.available_beds = new_beds
        h.doctors_available = new_doctors
        h.icu_available = new_icu
        st.success(f"Updated {h.name} successfully!")
        st.rerun()


# ═══════════════════════════════════════════
# TAB 3: SYSTEM ADMINISTRATION
# ═══════════════════════════════════════════
def system_admin_tab():
    """System Administrator interface."""
    st.title("⚙️ System Administration")
    st.markdown("*Manage dataset and traffic conditions.*")

    city = st.session_state.city_graph

    # ─── City Graph ───
    st.subheader("City Graph")
    hospitals = st.session_state.hospitals
    ambulances = st.session_state.ambulances
    if city.is_real_map():
        show_map(build_map_html(city, hospitals, ambulances), 560)
    else:
        fig = draw_city_graph(city, hospitals, ambulances,
                              title="City Road Network with Traffic")
        show_fig(fig)

    st.markdown("---")

    # ─── Ambulance Fleet & State Reset ───
    st.subheader("🚑 Ambulance Fleet")
    for amb in ambulances:
        st.text(f"{amb.ambulance_id} @ {city.label(amb.current_node)}: "
                f"{'Available' if amb.is_available else 'Busy / Dispatched'}")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        if st.button("✅ Release All Ambulances"):
            for amb in ambulances:
                amb.release()
            st.success("All ambulances released.")
            st.rerun()
    with fcol2:
        if st.button("♻️ Reset Fleet & Hospitals to Defaults"):
            _, fresh_hospitals, fresh_ambulances, _, _ = build_world(
                st.session_state.map_mode)
            st.session_state.hospitals = fresh_hospitals
            st.session_state.ambulances = fresh_ambulances
            st.session_state.dispatch_result = None
            st.success("Fleet and hospital resources reset.")
            st.rerun()

    st.markdown("---")

    # ─── Traffic Management ───
    st.subheader("🚦 Traffic Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔀 Randomize All Traffic", type="primary"):
            randomize_traffic(city)
            st.success("Traffic conditions randomized!")
            st.rerun()

    with col2:
        if st.button("🔄 Reset All Traffic (Clear Roads)"):
            for u, v, data in city.get_edges():
                city.update_traffic(u, v, 1.0)
            st.success("All traffic reset to clear!")
            st.rerun()

    # Manual traffic updates
    if city.is_real_map():
        st.markdown("**Update Traffic on a Named Road (all its segments):**")
        by_name = {}
        for u, v, d in city.get_edges():
            if d.get('name'):
                by_name.setdefault(d['name'], []).append((u, v))
        road_names = sorted(by_name)
        if road_names:
            road = st.selectbox("Select Road", road_names)
            segs = by_name[road]
            current_tf = round(sum(city.graph[u][v]['traffic_factor']
                                   for u, v in segs) / len(segs), 1)
            new_tf = st.slider(f"Traffic Factor ({road}, {len(segs)} segments)",
                               1.0, 3.0, min(max(current_tf, 1.0), 3.0), 0.1)
            if st.button("Update Traffic"):
                for u, v in segs:
                    city.update_traffic(u, v, new_tf)
                st.success(f"Updated traffic on {road} to {new_tf}x")
                st.rerun()
    else:
        st.markdown("**Update Individual Road Traffic:**")
        edges = [(u, v) for u, v, _ in city.get_edges()]
        edge_labels = [f"{u} ↔ {v}" for u, v in edges]

        selected_edge = st.selectbox("Select Road", edge_labels)
        if selected_edge:
            idx = edge_labels.index(selected_edge)
            u, v = edges[idx]
            current_tf = city.graph[u][v]['traffic_factor']
            new_tf = st.slider(f"Traffic Factor ({u} ↔ {v})", 1.0, 3.0, current_tf, 0.1)
            if st.button("Update Traffic"):
                city.update_traffic(u, v, new_tf)
                st.success(f"Updated traffic on {u} ↔ {v} to {new_tf}x")
                st.rerun()

    st.markdown("---")

    # ─── Enhanced Routing: Fuzzy Logic & Bayesian Reasoning ───
    st.subheader("🧠 AI-Enhanced Routing")

    with st.expander("⚙️ Fuzzy Logic & Bayesian Reasoning Settings", expanded=False):
        st.markdown(
            "*These features enhance the A\\* routing cost model with AI-based "
            "congestion estimation and road-risk analysis.*"
        )

        er_col1, er_col2 = st.columns(2)
        with er_col1:
            fuzzy_enabled = st.checkbox(
                "Enable Fuzzy Logic (Congestion)",
                value=city._fuzzy_enabled,
                help="Uses Mamdani fuzzy inference to estimate traffic congestion "
                     "severity from the traffic factor, replacing the raw multiplier "
                     "with a fuzzy-derived cost factor."
            )
        with er_col2:
            bayesian_enabled = st.checkbox(
                "Enable Bayesian Reasoning (Road Risk)",
                value=city._bayesian_enabled,
                help="Uses a Bayesian network to estimate the probability of road "
                     "blockage given weather, accident, and traffic evidence, adding "
                     "a risk-based cost penalty."
            )

        # Bayesian evidence and penalty controls (shown only when Bayesian is enabled)
        global_evidence = {}
        bayesian_penalty = city._bayesian_penalty_weight

        if bayesian_enabled:
            st.markdown("**Bayesian Evidence (simulated conditions):**")
            ev_col1, ev_col2, ev_col3 = st.columns(3)
            with ev_col1:
                rain = st.checkbox(
                    "🌧️ Rain",
                    value=city._global_evidence.get('rain', False),
                    help="Simulated weather input. Not from a real data source."
                )
                if rain:
                    global_evidence['rain'] = True
                else:
                    global_evidence['rain'] = False
            with ev_col2:
                accident = st.checkbox(
                    "🚗 Accident Reported",
                    value=city._global_evidence.get('accident', False),
                    help="Simulated accident report. Applies globally to all roads."
                )
                if accident:
                    global_evidence['accident'] = True
                else:
                    global_evidence['accident'] = False
            with ev_col3:
                bayesian_penalty = st.slider(
                    "Risk Penalty Weight",
                    0.0, 3.0,
                    float(city._bayesian_penalty_weight),
                    0.1,
                    help="Maximum additional cost multiplier when P(blocked)=1.0. "
                         "For example, 1.0 means a fully-blocked road doubles the cost."
                )

        # Apply configuration to city graph
        city.set_enhanced_routing(
            fuzzy_enabled=fuzzy_enabled,
            bayesian_enabled=bayesian_enabled,
            bayesian_penalty_weight=bayesian_penalty,
            global_evidence=global_evidence,
        )

        # Show current status
        if fuzzy_enabled or bayesian_enabled:
            status_parts = []
            if fuzzy_enabled:
                status_parts.append("**Fuzzy Logic**: ON")
            if bayesian_enabled:
                status_parts.append("**Bayesian Reasoning**: ON")
                if global_evidence.get('rain'):
                    status_parts.append("🌧️ Rain: Yes")
                if global_evidence.get('accident'):
                    status_parts.append("🚗 Accident: Yes")
                status_parts.append(f"Risk Penalty: {bayesian_penalty:.1f}×")
            st.info(" | ".join(status_parts))
        else:
            st.caption("Both features disabled — using original routing costs.")

    # ─── Live Model Demonstrator & Inspector ───
    with st.expander("🔬 Live AI Model Inspector & Demonstrator", expanded=True):
        st.markdown(
            "*Interactive tool to inspect and test how Fuzzy Logic and Bayesian Reasoning "
            "calculate scores, probabilities, and cost factors in real time.*"
        )

        demo_tab1, demo_tab2, demo_tab3 = st.tabs([
            "🧠 Fuzzy Logic Inspector",
            "🎲 Bayesian Risk Inspector",
            "⚖️ Road Cost Comparison"
        ])

        with demo_tab1:
            st.markdown("##### 🧠 Mamdani Fuzzy Inference Engine")
            st.caption(
                "Converts traffic factors into a continuous congestion score [0–100] "
                "via triangular membership functions and centroid defuzzification."
            )
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                test_tf = st.slider("Traffic Factor (Road Multiplier)", 1.0, 3.0, 2.2, 0.1, key="demo_fuzzy_tf")
                test_speed = st.slider(
                    "Simulated Speed Ratio (1.0 = Free Flow, 0.0 = Jam)",
                    0.0, 1.0,
                    float(round(1.0 / test_tf, 2)),
                    0.05,
                    key="demo_fuzzy_sr",
                    help="Simulated speed input derived from traffic factor (1.0 / traffic_factor)."
                )
            with f_col2:
                f_score = compute_fuzzy_congestion(test_tf, test_speed)
                f_factor = congestion_score_to_factor(f_score)

                if f_score < 30:
                    badge = "🟢 Low Congestion"
                elif f_score < 60:
                    badge = "🟡 Moderate Congestion"
                elif f_score < 80:
                    badge = "🟠 Heavy Congestion"
                else:
                    badge = "🔴 Severe / Standstill"

                st.metric("Congestion Score (0–100)", f"{f_score:.1f}", badge)
                st.metric("Fuzzy Cost Multiplier", f"{f_factor:.2f}×",
                          f"Replaces raw {test_tf:.1f}× multiplier")
                st.progress(min(max(f_score / 100.0, 0.0), 1.0))

        with demo_tab2:
            st.markdown("##### 🎲 Bayesian Network Risk Engine")
            st.caption(
                "Uses exact enumeration over CPTs to compute P(Road_Blocked | evidence), "
                "marginalizing over any unobserved variables."
            )
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                b_rain = st.checkbox("🌧️ Weather: Rain present", value=True, key="demo_b_rain")
                b_acc = st.checkbox("🚗 Incident: Accident reported", value=False, key="demo_b_acc")
                b_tl = st.selectbox(
                    "Observed Traffic Level",
                    ["low", "medium", "high"],
                    index=1,
                    key="demo_b_tl"
                )
                b_penalty = st.slider(
                    "Risk Penalty Weight",
                    0.0, 3.0, 1.0, 0.1,
                    key="demo_b_pw",
                    help="Multiplier applied to blockage probability: Cost × (1 + penalty × P)"
                )
            with b_col2:
                b_prob = compute_bayesian_risk(2.0, rain=b_rain, accident=b_acc, traffic_level=b_tl)
                b_factor = risk_probability_to_factor(b_prob, b_penalty)

                if b_prob < 0.20:
                    risk_badge = "🟢 Low Risk"
                elif b_prob < 0.50:
                    risk_badge = "🟡 Moderate Risk"
                else:
                    risk_badge = "🔴 High Risk"

                st.metric("P(Road Blocked | Evidence)", f"{b_prob * 100:.1f}%", risk_badge)
                st.metric("Bayesian Risk Multiplier", f"{b_factor:.2f}×",
                          f"+{(b_factor - 1.0) * 100:.1f}% additional risk cost")
                st.progress(min(max(b_prob, 0.0), 1.0))

        with demo_tab3:
            st.markdown("##### ⚖️ Road Cost Transformation on a 10 km Road")
            st.caption(
                "Comparison showing how baseline distance × traffic_factor is modified "
                "by Fuzzy Logic (congestion smoothing) and Bayesian Reasoning (risk penalty)."
            )
            comp_res = compute_effective_weight(
                base_distance=10.0,
                traffic_factor=test_tf,
                fuzzy_enabled=True,
                bayesian_enabled=True,
                bayesian_evidence={'rain': b_rain, 'accident': b_acc, 'traffic_level': b_tl},
                bayesian_penalty_weight=b_penalty
            )
            base_cost = 10.0 * test_tf
            fuzzy_cost = 10.0 * comp_res['congestion_factor']
            bayesian_cost = 10.0 * test_tf * comp_res['risk_factor']
            total_cost = comp_res['weight']

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Cost", f"{base_cost:.1f}")
            c2.metric("With Fuzzy Only", f"{fuzzy_cost:.1f}",
                      delta=f"{fuzzy_cost - base_cost:+.1f}")
            c3.metric("With Bayesian Only", f"{bayesian_cost:.1f}",
                      delta=f"{bayesian_cost - base_cost:+.1f}")
            c4.metric("Combined AI Cost", f"{total_cost:.1f}",
                      delta=f"{total_cost - base_cost:+.1f}")


if __name__ == "__main__":
    main()

