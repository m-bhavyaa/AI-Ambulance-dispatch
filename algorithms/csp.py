"""
Constraint Satisfaction Problem (CSP) Implementation
Selects a hospital for an emergency using constraint filtering followed by a
backtracking value search ordered by road-network cost.

Variable : the hospital assigned to this emergency.
Domain   : all hospitals.

Hard constraints (applied in sequence, each one is a domain reduction):
    C1  available_beds > 0
    C2  icu_available            (only if ICU is required)
    C3  doctors_available > 0
    C4  occupied_beds < capacity_limit
    C5  the hospital is reachable by road from the emergency location

Value ordering / tie-breaking:
    Remaining hospitals are ordered by the A* route cost from the emergency
    (the same effective weights used for routing: traffic, fuzzy, Bayesian).
    Euclidean straight-line distance is used only as a tie-breaker, then the
    hospital name for full determinism.

The search tries values in that order and takes the first one that is
consistent with every constraint. If a candidate fails its final check the
search backtracks to the next one (backtrack_count counts those events).
"""

from algorithms.astar import astar_search


def hospital_csp(hospitals, icu_required, emergency_node, city_graph):
    """
    Solve hospital selection as a CSP.

    Args:
        hospitals (list[Hospital]): All hospitals.
        icu_required (bool): Whether an ICU is required.
        emergency_node (str): The emergency location node.
        city_graph (CityGraph): City graph (routing + distance).

    Returns:
        dict: {
            'assigned_hospital': Hospital or None,
            'steps': list of step dicts for visualization,
            'domain_reductions': domain after each constraint,
            'backtrack_count': number of backtracks during value search,
            'constraints': list of constraint descriptions,
            'route_results': {hospital name: A* result from emergency},
            'route_costs': {hospital name: route cost},
            'unreachable': names of hospitals with no road route,
            'ranking': [(name, route_cost, euclidean)] in selection order,
            'failure_reason': None or a human-readable explanation,
        }
    """
    steps = []
    domain_reductions = []
    backtrack_count = 0

    constraints = [
        "C1: Hospital must have available beds (available_beds > 0)",
        "C2: ICU must be available (if required)",
        "C3: Doctor must be available (doctors_available > 0)",
        "C4: Capacity must not exceed limit (occupied_beds < capacity_limit)",
        "C5: Hospital must be reachable by road from the emergency location",
    ]

    def add_step(action, description, domain, status, **extra):
        step = {
            'step': len(steps) + 1,
            'action': action,
            'description': description,
            'domain': list(domain),
            'status': status,
        }
        step.update(extra)
        steps.append(step)

    def result(assigned, failure_reason=None, route_results=None,
               unreachable=None, ranking=None):
        route_results = route_results or {}
        return {
            'assigned_hospital': assigned,
            'steps': steps,
            'domain_reductions': domain_reductions,
            'backtrack_count': backtrack_count,
            'constraints': constraints,
            'route_results': route_results,
            'route_costs': {n: r['cost'] for n, r in route_results.items()},
            'unreachable': unreachable or [],
            'ranking': ranking or [],
            'failure_reason': failure_reason,
        }

    # ── Initial domain ──
    initial_names = [h.name for h in hospitals]
    add_step('Initialize Domain',
             f'Initial domain contains all {len(hospitals)} hospitals',
             initial_names, 'info')
    domain_reductions.append({
        'stage': 'Initial',
        'domain': list(initial_names),
        'removed': [],
        'reason': 'All hospitals in domain',
    })

    current = list(hospitals)

    def apply_constraint(label, stage, reason, predicate, description):
        """Filter `current` by predicate and record the reduction."""
        nonlocal current
        kept, removed = [], []
        for h in current:
            (kept if predicate(h) else removed).append(h)
        current = kept
        names = [h.name for h in kept]
        removed_names = [h.name for h in removed]
        add_step(label,
                 f'{description} Removed: {removed_names if removed_names else "None"}',
                 names, 'constraint_check', removed=removed_names)
        domain_reductions.append({
            'stage': stage,
            'domain': list(names),
            'removed': removed_names,
            'reason': reason,
        })

    # ── C1: beds ──
    apply_constraint('Apply C1: Bed Availability', 'After C1 (Beds)',
                     'No available beds',
                     lambda h: h.available_beds > 0,
                     'Checking available_beds > 0.')

    # ── C2: ICU (only if required) ──
    if icu_required:
        apply_constraint('Apply C2: ICU Availability', 'After C2 (ICU)',
                         'ICU not available',
                         lambda h: h.icu_available,
                         'ICU required. Checking icu_available == True.')
    else:
        names = [h.name for h in current]
        add_step('Skip C2: ICU Not Required',
                 'ICU not required for this emergency. Constraint skipped.',
                 names, 'skipped', removed=[])
        domain_reductions.append({
            'stage': 'After C2 (ICU - Skipped)',
            'domain': list(names),
            'removed': [],
            'reason': 'ICU not required',
        })

    # ── C3: doctors ──
    apply_constraint('Apply C3: Doctor Availability', 'After C3 (Doctors)',
                     'No doctors available',
                     lambda h: h.doctors_available > 0,
                     'Checking doctors_available > 0.')

    # ── C4: capacity limit ──
    apply_constraint('Apply C4: Capacity Limit', 'After C4 (Capacity)',
                     'Capacity limit reached',
                     lambda h: (h.total_beds - h.available_beds) < h.capacity_limit,
                     'Checking occupied_beds < capacity_limit.')

    # ── C5: road reachability (routes are computed once and reused) ──
    route_results = {}
    for h in current:
        route = astar_search(city_graph, emergency_node, h.node)
        if route is not None:
            route_results[h.name] = route
    apply_constraint('Apply C5: Road Reachability', 'After C5 (Reachable)',
                     'No road route from emergency',
                     lambda h: h.name in route_results,
                     f'Running A* from {emergency_node} to each remaining hospital.')
    unreachable = list(domain_reductions[-1]['removed'])

    # ── Empty domain: explain why ──
    if not current:
        reason = _explain_failure(domain_reductions)
        add_step('Backtracking: No Solution',
                 f'Domain is empty after constraint propagation. {reason}',
                 [], 'failure')
        return result(None, failure_reason=reason, route_results={},
                      unreachable=unreachable)

    # ── Value ordering: road cost first, Euclidean distance as tie-breaker ──
    ranking = sorted(
        ((h, route_results[h.name]['cost'],
          city_graph.euclidean_distance(emergency_node, h.node))
         for h in current),
        key=lambda t: (round(t[1], 9), t[2], t[0].name),
    )
    ordered_names = [h.name for h, _, _ in ranking]
    add_step('Order Values by Road Cost',
             'Order: ' + ', '.join(
                 f'{h.name} (route {c:.2f}, straight-line {e:.2f})'
                 for h, c, e in ranking),
             ordered_names, 'heuristic')

    # ── Backtracking value search ──
    assigned = None
    for h, cost, euclid in ranking:
        add_step(f'Backtracking: Try {h.name}',
                 f'Testing {h.name} (node {h.node}): route cost={cost:.2f}, '
                 f'beds={h.available_beds}, ICU={"Yes" if h.icu_available else "No"}, '
                 f'doctors={h.doctors_available}',
                 [h.name], 'trying')

        consistent = (
            h.available_beds > 0
            and (not icu_required or h.icu_available)
            and h.doctors_available > 0
            and (h.total_beds - h.available_beds) < h.capacity_limit
            and h.name in route_results
        )
        if consistent:
            assigned = h
            break

        backtrack_count += 1
        add_step(f'Backtrack from {h.name}',
                 f'{h.name} violates a constraint. Trying next value.',
                 [h.name], 'backtrack')

    ranking_out = [(h.name, c, e) for h, c, e in ranking]
    if assigned is not None:
        cost = route_results[assigned.name]['cost']
        add_step('Assignment Complete',
                 f'Assigned hospital: {assigned.name} at node {assigned.node} '
                 f'(route cost: {cost:.2f})',
                 [assigned.name], 'success')
        return result(assigned, route_results=route_results,
                      unreachable=unreachable, ranking=ranking_out)

    add_step('Backtracking: No Solution',
             'All candidate hospitals were rejected.', [], 'failure')
    return result(None, failure_reason='All candidate hospitals were rejected.',
                  route_results=route_results, unreachable=unreachable,
                  ranking=ranking_out)


def _explain_failure(domain_reductions):
    """Describe which constraint emptied the domain."""
    if domain_reductions and not domain_reductions[0]['domain']:
        return 'No hospitals are registered in the system.'
    for dr in domain_reductions[1:]:
        if not dr['domain']:
            return f"No hospital remains after {dr['stage']}: {dr['reason'].lower()}."
    return 'No hospital satisfies all constraints.'
