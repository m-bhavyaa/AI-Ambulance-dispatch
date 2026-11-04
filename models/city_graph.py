"""
City Graph Model
Builds and manages the city road network using NetworkX.
Nodes = intersections, Edges = roads with distance + traffic weights.
"""

import networkx as nx
import math


class CityGraph:
    """
    Represents the city road network as a weighted graph.
    Nodes are intersections with (x, y) positions.
    Edges are roads with base distance and traffic factor.
    """

    def __init__(self):
        """Initialize an empty city graph."""
        self.graph = nx.Graph()
        # Enhanced routing configuration (disabled by default)
        self._fuzzy_enabled = False
        self._bayesian_enabled = False
        self._bayesian_penalty_weight = 1.0
        self._global_evidence = {}  # Global Bayesian evidence (e.g., rain)

    def add_intersection(self, node_id, x, y, **attrs):
        """
        Add an intersection (node) to the city graph.

        Args:
            node_id (str): Unique identifier for the intersection.
            x (float): X coordinate.
            y (float): Y coordinate.
            **attrs: Optional extra attributes (e.g. lat, lon, label for the
                real-map mode). The abstract demo city passes none.
        """
        self.graph.add_node(node_id, pos=(x, y), **attrs)

    def add_road(self, node1, node2, traffic_factor=1.0, base_distance=None, **attrs):
        """
        Add a road (edge) between two intersections.
        By default the base weight is the Euclidean distance between nodes.
        The effective weight = base_distance * traffic_factor.

        Args:
            node1 (str): First intersection.
            node2 (str): Second intersection.
            traffic_factor (float): Traffic multiplier (1.0 = clear, up to 3.0 = heavy).
            base_distance (float): Optional explicit road length (same units as
                the node coordinates). Real roads curve, so this must be >= the
                straight-line distance to keep the A* heuristic admissible.
            **attrs: Optional extra edge attributes (road name, geometry, ...).
        """
        pos1 = self.graph.nodes[node1]['pos']
        pos2 = self.graph.nodes[node2]['pos']
        if base_distance is None:
            base_distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        weight = base_distance * traffic_factor
        self.graph.add_edge(node1, node2,
                            base_distance=base_distance,
                            traffic_factor=traffic_factor,
                            weight=weight,
                            **attrs)

    def label(self, node):
        """Human-readable name of a node (falls back to the node id)."""
        return self.graph.nodes[node].get('label', node)

    def get_selectable_nodes(self):
        """
        Nodes a user may pick as an emergency location. Real maps flag true
        junctions with is_junction; the abstract demo city offers every node.
        """
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get('is_junction', True)]
        return nodes or list(self.graph.nodes())

    def is_real_map(self):
        """True when nodes carry real-world lat/lon (Dwarka mode)."""
        return any('lat' in d for _, d in self.graph.nodes(data=True))

    def update_traffic(self, node1, node2, new_traffic_factor):
        """
        Update the traffic factor on a road and recalculate weight.

        Args:
            node1 (str): First intersection.
            node2 (str): Second intersection.
            new_traffic_factor (float): New traffic multiplier.
        """
        if self.graph.has_edge(node1, node2):
            edge = self.graph[node1][node2]
            edge['traffic_factor'] = new_traffic_factor
            edge['weight'] = edge['base_distance'] * new_traffic_factor

    def get_position(self, node):
        """Get the (x, y) position of a node."""
        return self.graph.nodes[node]['pos']

    def get_all_positions(self):
        """Get positions of all nodes as a dict {node: (x, y)}."""
        return nx.get_node_attributes(self.graph, 'pos')

    def euclidean_distance(self, node1, node2):
        """
        Calculate Euclidean distance between two nodes.
        Used as the heuristic h(n) in A* search.

        Args:
            node1 (str): First node.
            node2 (str): Second node.

        Returns:
            float: Euclidean distance.
        """
        pos1 = self.graph.nodes[node1]['pos']
        pos2 = self.graph.nodes[node2]['pos']
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_neighbors(self, node):
        """
        Get all neighbors of a node with edge weights.

        When enhanced routing is enabled (fuzzy and/or Bayesian), the weights
        are computed through the cost integration pipeline. Otherwise, returns
        the original base_distance * traffic_factor weights.
        """
        if not (self._fuzzy_enabled or self._bayesian_enabled):
            # Original behavior — no change
            neighbors = []
            for neighbor in self.graph.neighbors(node):
                weight = self.graph[node][neighbor]['weight']
                neighbors.append((neighbor, weight))
            return neighbors

        # Enhanced routing: compute effective weights via cost integration
        from algorithms.cost_integration import compute_effective_weight

        neighbors = []
        for neighbor in self.graph.neighbors(node):
            edge = self.graph[node][neighbor]
            base_distance = edge['base_distance']
            traffic_factor = edge['traffic_factor']

            # Merge global evidence with any per-edge evidence
            evidence = dict(self._global_evidence)
            edge_evidence = edge.get('bayesian_evidence', {})
            if edge_evidence:
                evidence.update(edge_evidence)

            result = compute_effective_weight(
                base_distance=base_distance,
                traffic_factor=traffic_factor,
                fuzzy_enabled=self._fuzzy_enabled,
                bayesian_enabled=self._bayesian_enabled,
                bayesian_evidence=evidence,
                bayesian_penalty_weight=self._bayesian_penalty_weight,
            )
            neighbors.append((neighbor, result['weight']))
        return neighbors

    def set_enhanced_routing(self, fuzzy_enabled=False, bayesian_enabled=False,
                             bayesian_penalty_weight=1.0, global_evidence=None):
        """
        Configure enhanced routing features (fuzzy logic and Bayesian reasoning).

        When both features are disabled, get_neighbors() returns the original
        base_distance * traffic_factor weights, preserving backward compatibility.

        Args:
            fuzzy_enabled (bool): Enable fuzzy congestion adjustment.
            bayesian_enabled (bool): Enable Bayesian risk adjustment.
            bayesian_penalty_weight (float): Max cost penalty at P(blocked)=1.0.
            global_evidence (dict): Global Bayesian evidence applied to all edges.
                Keys: 'rain' (bool), 'accident' (bool), 'traffic_level' (str).
        """
        self._fuzzy_enabled = bool(fuzzy_enabled)
        self._bayesian_enabled = bool(bayesian_enabled)
        self._bayesian_penalty_weight = max(0.0, float(bayesian_penalty_weight))
        self._global_evidence = dict(global_evidence) if global_evidence else {}

    def set_edge_evidence(self, node1, node2, evidence):
        """
        Set Bayesian evidence for a specific road (edge).

        Per-edge evidence overrides global evidence for matching keys.
        For example, setting accident=True on a specific road.

        Args:
            node1 (str): First intersection.
            node2 (str): Second intersection.
            evidence (dict): Evidence dict, e.g. {'accident': True}.
        """
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2]['bayesian_evidence'] = dict(evidence)

    def clear_edge_evidence(self, node1, node2):
        """Clear Bayesian evidence for a specific road."""
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2].pop('bayesian_evidence', None)

    def get_enhanced_routing_config(self):
        """
        Get the current enhanced routing configuration.

        Returns:
            dict: Current configuration state.
        """
        return {
            'fuzzy_enabled': self._fuzzy_enabled,
            'bayesian_enabled': self._bayesian_enabled,
            'bayesian_penalty_weight': self._bayesian_penalty_weight,
            'global_evidence': dict(self._global_evidence),
        }

    def get_nodes(self):
        """Get all nodes in the graph."""
        return list(self.graph.nodes())

    def get_edges(self):
        """Get all edges with their data."""
        return list(self.graph.edges(data=True))
