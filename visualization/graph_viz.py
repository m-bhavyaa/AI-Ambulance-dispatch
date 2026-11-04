"""
Graph Visualization Module
Visualizes the city graph, A* search exploration, and optimal paths
using Matplotlib and NetworkX.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def draw_city_graph(city_graph, hospitals=None, ambulances=None,
                    emergency_node=None, title="City Road Network"):
    """
    Draw the city graph showing intersections, roads, hospitals, and ambulances.

    Args:
        city_graph (CityGraph): The city graph to draw.
        hospitals (list[Hospital]): Hospitals to mark on the graph.
        ambulances (list[Ambulance]): Ambulances to mark on the graph.
        emergency_node (str): Emergency location node to highlight.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    G = city_graph.graph
    pos = city_graph.get_all_positions()

    # Categorize nodes
    hospital_nodes = [h.node for h in hospitals] if hospitals else []
    ambulance_nodes = [a.current_node for a in ambulances if a.is_available] if ambulances else []

    # Draw edges with traffic coloring
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        tf = data.get('traffic_factor', 1.0)
        if tf <= 1.5:
            edge_colors.append('#2ecc71')  # Green - clear
        elif tf <= 2.0:
            edge_colors.append('#f39c12')  # Orange - moderate
        else:
            edge_colors.append('#e74c3c')  # Red - heavy
        edge_widths.append(2.0)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)

    # Draw edge labels (traffic factor)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        tf = data.get('traffic_factor', 1.0)
        if tf > 1.0:
            edge_labels[(u, v)] = f"{tf:.1f}x"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # Color nodes by type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if emergency_node and node == emergency_node:
            node_colors.append('#e74c3c')  # Red - emergency
            node_sizes.append(500)
        elif node in hospital_nodes:
            node_colors.append('#2ecc71')  # Green - hospital
            node_sizes.append(400)
        elif node in ambulance_nodes:
            node_colors.append('#3498db')  # Blue - ambulance
            node_sizes.append(400)
        else:
            node_colors.append('#95a5a6')  # Gray - regular
            node_sizes.append(200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='#2c3e50', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold',
                            font_color='white', ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#95a5a6', label='Intersection'),
        mpatches.Patch(color='#2ecc71', label='Hospital'),
        mpatches.Patch(color='#3498db', label='Ambulance'),
        mpatches.Patch(color='#e74c3c', label='Emergency'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.axis('off')
    plt.tight_layout()

    return fig


def draw_astar_exploration(city_graph, astar_result, hospitals=None,
                           ambulances=None, emergency_node=None,
                           ambulance_node=None):
    """
    Draw the A* search exploration showing explored nodes and final path.

    Args:
        city_graph (CityGraph): The city graph.
        astar_result (dict): Result from astar_search().
        hospitals (list[Hospital]): Hospitals to mark.
        ambulances (list[Ambulance]): Ambulances to mark.
        emergency_node (str): Emergency location.
        ambulance_node (str): Selected ambulance location.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    G = city_graph.graph
    pos = city_graph.get_all_positions()

    hospital_nodes = [h.node for h in hospitals] if hospitals else []

    # --- Left plot: Node Exploration ---
    ax1 = axes[0]
    explored = astar_result['explored_nodes']
    path = astar_result['path']

    # Draw all edges gray
    nx.draw_networkx_edges(G, pos, edge_color='#bdc3c7', width=1.5, ax=ax1)

    # Highlight explored edges
    explored_edges = astar_result['explored_edges']
    if explored_edges:
        nx.draw_networkx_edges(G, pos, edgelist=explored_edges,
                               edge_color='#f39c12', width=2.5, ax=ax1,
                               style='dashed')

    # Color nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == emergency_node:
            node_colors.append('#e74c3c')
            node_sizes.append(500)
        elif node == ambulance_node:
            node_colors.append('#3498db')
            node_sizes.append(500)
        elif node in hospital_nodes:
            node_colors.append('#2ecc71')
            node_sizes.append(400)
        elif node in explored:
            node_colors.append('#f39c12')  # Yellow - explored
            node_sizes.append(300)
        else:
            node_colors.append('#ecf0f1')
            node_sizes.append(200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='#2c3e50', linewidths=1.5, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax1)

    ax1.set_title(f"A* Node Exploration ({len(explored)} nodes explored)",
                  fontsize=12, fontweight='bold')
    ax1.set_facecolor('#f8f9fa')
    ax1.axis('off')

    # --- Right plot: Final Optimal Path ---
    ax2 = axes[1]

    # Draw all edges gray
    nx.draw_networkx_edges(G, pos, edge_color='#bdc3c7', width=1.5, ax=ax2)

    # Highlight optimal path
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               edge_color='#e74c3c', width=4.0, ax=ax2)

    # Color nodes
    node_colors2 = []
    node_sizes2 = []
    for node in G.nodes():
        if node == emergency_node:
            node_colors2.append('#e74c3c')
            node_sizes2.append(500)
        elif node == ambulance_node:
            node_colors2.append('#3498db')
            node_sizes2.append(500)
        elif node in hospital_nodes:
            node_colors2.append('#2ecc71')
            node_sizes2.append(400)
        elif node in path:
            node_colors2.append('#e74c3c')
            node_sizes2.append(350)
        else:
            node_colors2.append('#ecf0f1')
            node_sizes2.append(200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors2, node_size=node_sizes2,
                           edgecolors='#2c3e50', linewidths=1.5, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax2)

    ax2.set_title(f"Final Optimal Path (cost: {astar_result['cost']:.2f})",
                  fontsize=12, fontweight='bold')
    ax2.set_facecolor('#f8f9fa')
    ax2.axis('off')

    # Legends
    legend1 = [
        mpatches.Patch(color='#f39c12', label='Explored Nodes'),
        mpatches.Patch(color='#e74c3c', label='Emergency'),
        mpatches.Patch(color='#3498db', label='Ambulance'),
        mpatches.Patch(color='#2ecc71', label='Hospital'),
    ]
    ax1.legend(handles=legend1, loc='upper left', fontsize=8)

    legend2 = [
        mpatches.Patch(color='#e74c3c', label='Optimal Path'),
        mpatches.Patch(color='#3498db', label='Ambulance'),
        mpatches.Patch(color='#2ecc71', label='Hospital'),
    ]
    ax2.legend(handles=legend2, loc='upper left', fontsize=8)

    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()

    return fig


def draw_comparison(city_graph, astar_result, dijkstra_result,
                    ambulance_node, emergency_node, hospital_node):
    """
    Draw side-by-side comparison of A* vs Dijkstra search.

    Args:
        city_graph (CityGraph): The city graph.
        astar_result (dict): A* search result.
        dijkstra_result (dict): Dijkstra search result.
        ambulance_node (str): Ambulance start node.
        emergency_node (str): Emergency node.
        hospital_node (str): Hospital goal node.

    Returns:
        matplotlib.figure.Figure: The comparison figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    G = city_graph.graph
    pos = city_graph.get_all_positions()

    for ax, result, title in zip(
        axes,
        [astar_result, dijkstra_result],
        ["A* Search", "Dijkstra (No Heuristic)"]
    ):
        explored = result['explored_nodes']
        path = result['path']

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='#bdc3c7', width=1.5, ax=ax)

        # Highlight explored
        explored_edges = result['explored_edges']
        if explored_edges:
            nx.draw_networkx_edges(G, pos, edgelist=explored_edges,
                                   edge_color='#f39c12', width=2.0, ax=ax,
                                   style='dashed')

        # Highlight path
        if path:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                   edge_color='#e74c3c', width=3.5, ax=ax)

        # Node colors
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == emergency_node or node == ambulance_node:
                node_colors.append('#3498db')
                node_sizes.append(450)
            elif node == hospital_node:
                node_colors.append('#2ecc71')
                node_sizes.append(450)
            elif node in path:
                node_colors.append('#e74c3c')
                node_sizes.append(300)
            elif node in explored:
                node_colors.append('#f39c12')
                node_sizes.append(250)
            else:
                node_colors.append('#ecf0f1')
                node_sizes.append(200)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               edgecolors='#2c3e50', linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

        ax.set_title(
            f"{title}\nNodes Expanded: {result['nodes_expanded']} | "
            f"Path Cost: {result['cost']:.2f} | "
            f"Time: {result['time_taken']*1000:.2f}ms",
            fontsize=11, fontweight='bold'
        )
        ax.set_facecolor('#f8f9fa')
        ax.axis('off')

    fig.suptitle("Performance Comparison: A* Search vs Dijkstra",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()

    return fig
