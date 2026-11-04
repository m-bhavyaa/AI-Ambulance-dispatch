"""
A* Search Algorithm Implementation
Finds the shortest path between two nodes in a weighted graph
using f(n) = g(n) + h(n) where h(n) is Euclidean distance.
Also includes simple Dijkstra search for performance comparison.
"""

import heapq
import time
from collections import deque


def astar_search(city_graph, start, goal):
    """
    A* Search Algorithm to find optimal path from start to goal.

    Uses f(n) = g(n) + h(n) where:
    - g(n) = actual cost from start to current node (distance * traffic)
    - h(n) = Euclidean distance from current node to goal (heuristic)

    Args:
        city_graph (CityGraph): The city road network graph.
        start (str): Starting node.
        goal (str): Goal node.

    Returns:
        dict: {
            'path': list of nodes in optimal path,
            'cost': total path cost,
            'explored_nodes': list of nodes in exploration order,
            'explored_edges': list of (from, to) edges explored,
            'frontier_history': list of frontier states at each step,
            'g_scores': dict of final g-scores for explored nodes,
            'f_scores': dict of final f-scores for explored nodes,
            'steps': list of step-by-step details for visualization,
            'time_taken': execution time in seconds,
            'nodes_expanded': number of nodes expanded,
        }
        Returns None if no path found.
    """
    start_time = time.time()

    # Priority queue: (f_score, counter, node)
    # Counter is used to break ties
    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start))

    # Track where we came from for path reconstruction
    came_from = {}

    # g_score: actual cost from start to node
    g_score = {start: 0}

    # f_score: g_score + heuristic
    f_score = {start: city_graph.euclidean_distance(start, goal)}

    # Track explored nodes in order
    explored_nodes = []
    explored_edges = []
    steps = []
    open_set_nodes = {start}
    closed = set()

    while open_set:
        # Get node with lowest f_score
        current_f, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        open_set_nodes.discard(current)

        # Record step details
        step_info = {
            'current_node': current,
            'g_score': g_score.get(current, float('inf')),
            'f_score': current_f,
            'h_score': current_f - g_score.get(current, 0),
            'action': 'expand',
        }

        explored_nodes.append(current)

        # Goal reached
        if current == goal:
            step_info['action'] = 'goal_reached'
            steps.append(step_info)

            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            end_time = time.time()

            return {
                'path': path,
                'cost': g_score[goal],
                'explored_nodes': explored_nodes,
                'explored_edges': explored_edges,
                'g_scores': dict(g_score),
                'f_scores': dict(f_score),
                'steps': steps,
                'time_taken': end_time - start_time,
                'nodes_expanded': len(explored_nodes),
            }

        steps.append(step_info)

        # Explore neighbors
        for neighbor, weight in city_graph.get_neighbors(current):
            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, float('inf')):
                # Found a better path to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = city_graph.euclidean_distance(neighbor, goal)
                f_score[neighbor] = tentative_g + h

                explored_edges.append((current, neighbor))

                counter += 1
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                open_set_nodes.add(neighbor)

    end_time = time.time()
    return None  # No path found


def dijkstra_search(city_graph, start, goal):
    """
    Simple Dijkstra search (A* without heuristic) for performance comparison.
    Uses f(n) = g(n) only (no heuristic).

    Args:
        city_graph (CityGraph): The city road network graph.
        start (str): Starting node.
        goal (str): Goal node.

    Returns:
        dict: Same format as astar_search results.
    """
    start_time = time.time()

    open_set = []
    counter = 0
    heapq.heappush(open_set, (0, counter, start))

    came_from = {}
    g_score = {start: 0}
    explored_nodes = []
    explored_edges = []
    steps = []
    open_set_nodes = {start}
    closed = set()

    while open_set:
        current_g, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        open_set_nodes.discard(current)

        step_info = {
            'current_node': current,
            'g_score': g_score.get(current, float('inf')),
            'f_score': current_g,
            'h_score': 0,
            'action': 'expand',
        }

        explored_nodes.append(current)

        if current == goal:
            step_info['action'] = 'goal_reached'
            steps.append(step_info)

            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()

            end_time = time.time()

            return {
                'path': path,
                'cost': g_score[goal],
                'explored_nodes': explored_nodes,
                'explored_edges': explored_edges,
                'g_scores': dict(g_score),
                'f_scores': {k: v for k, v in g_score.items()},
                'steps': steps,
                'time_taken': end_time - start_time,
                'nodes_expanded': len(explored_nodes),
            }

        steps.append(step_info)

        for neighbor, weight in city_graph.get_neighbors(current):
            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                explored_edges.append((current, neighbor))

                counter += 1
                heapq.heappush(open_set, (g_score[neighbor], counter, neighbor))
                open_set_nodes.add(neighbor)

    end_time = time.time()
    return None


def compare_search_results(astar_result, dijkstra_result, tol=1e-9):
    """
    Compare an A* result with the Dijkstra baseline on the same graph.

    Both searches read the same effective edge weights from
    CityGraph.get_neighbors(), so for an admissible heuristic their optimal
    costs must be identical; A* should only differ in nodes expanded.

    Returns:
        dict with 'costs_match', 'astar_cost', 'dijkstra_cost',
        'astar_nodes_expanded', 'dijkstra_nodes_expanded', 'nodes_saved',
        'same_path', or None if either result is missing.
    """
    if astar_result is None or dijkstra_result is None:
        return None
    return {
        'costs_match': abs(astar_result['cost'] - dijkstra_result['cost']) <= tol,
        'astar_cost': astar_result['cost'],
        'dijkstra_cost': dijkstra_result['cost'],
        'astar_nodes_expanded': astar_result['nodes_expanded'],
        'dijkstra_nodes_expanded': dijkstra_result['nodes_expanded'],
        'nodes_saved': dijkstra_result['nodes_expanded'] - astar_result['nodes_expanded'],
        'same_path': astar_result['path'] == dijkstra_result['path'],
    }
