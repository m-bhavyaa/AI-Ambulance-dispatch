"""
CSP Visualization Module
Visualizes the CSP solving process: variables, domains,
constraint checking, domain reduction, backtracking, and final assignment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_csp_domain_reduction(csp_result):
    """
    Draw the CSP domain reduction process as a horizontal bar chart.
    Shows how the domain shrinks after each constraint is applied.

    Args:
        csp_result (dict): Result from hospital_csp().

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    domain_reductions = csp_result['domain_reductions']
    stages = [dr['stage'] for dr in domain_reductions]
    domain_sizes = [len(dr['domain']) for dr in domain_reductions]

    # Colors: green for large domain, yellow for medium, red for small
    colors = []
    max_size = max(domain_sizes + [1])
    for size in domain_sizes:
        ratio = size / max_size
        if ratio > 0.6:
            colors.append('#2ecc71')
        elif ratio > 0.3:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    y_pos = np.arange(len(stages))
    bars = ax.barh(y_pos, domain_sizes, color=colors, edgecolor='#2c3e50',
                   linewidth=1.2, height=0.6)

    # Add count labels on bars
    for bar, size, dr in zip(bars, domain_sizes, domain_reductions):
        label = ', '.join(dr['domain']) if dr['domain'] else 'Empty'
        if len(label) > 50:
            label = label[:47] + '...'
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f' {size} hospitals: {label}',
                va='center', fontsize=8, color='#2c3e50')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlabel('Number of Hospitals in Domain', fontsize=11)
    ax.set_title('CSP Domain Reduction Process', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_xlim(0, max(domain_sizes + [1]) + 3)
    plt.tight_layout()

    return fig


def draw_csp_backtracking(csp_result):
    """
    Draw the CSP backtracking process as a flowchart-style visualization.
    Shows each step of the backtracking search.

    Args:
        csp_result (dict): Result from hospital_csp().

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    steps = csp_result['steps']
    n_steps = len(steps)

    fig, ax = plt.subplots(figsize=(14, max(6, n_steps * 0.8)))

    # Draw each step as a box
    y_positions = list(range(n_steps))
    y_positions.reverse()

    for i, step in enumerate(steps):
        y = y_positions[i]
        status = step.get('status', 'info')

        # Choose color based on status
        if status == 'success':
            color = '#2ecc71'
            text_color = 'white'
        elif status == 'failure':
            color = '#e74c3c'
            text_color = 'white'
        elif status == 'backtrack':
            color = '#e67e22'
            text_color = 'white'
        elif status == 'constraint_check':
            color = '#3498db'
            text_color = 'white'
        elif status == 'heuristic':
            color = '#9b59b6'
            text_color = 'white'
        elif status == 'trying':
            color = '#f39c12'
            text_color = '#2c3e50'
        elif status == 'skipped':
            color = '#95a5a6'
            text_color = 'white'
        else:
            color = '#ecf0f1'
            text_color = '#2c3e50'

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (0.5, y - 0.35), 12, 0.7,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#2c3e50', linewidth=1.5
        )
        ax.add_patch(rect)

        # Step text
        step_text = f"Step {step['step']}: {step['action']}"
        ax.text(1.0, y + 0.1, step_text,
                fontsize=9, fontweight='bold', color=text_color, va='center')
        ax.text(1.0, y - 0.15, step['description'],
                fontsize=7.5, color=text_color, va='center', alpha=0.9)

        # Draw arrow to next step
        if i < n_steps - 1:
            next_y = y_positions[i + 1]
            ax.annotate('', xy=(6.5, next_y + 0.35), xytext=(6.5, y - 0.35),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    ax.set_xlim(0, 14)
    ax.set_ylim(-1, n_steps)
    ax.set_title('CSP Backtracking Search Process', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()

    return fig


def draw_csp_constraints(csp_result):
    """
    Draw the CSP constraints as a table visualization.

    Args:
        csp_result (dict): Result from hospital_csp().

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    constraints = csp_result['constraints']

    # Create table data
    table_data = []
    for i, constraint in enumerate(constraints):
        parts = constraint.split(': ', 1)
        c_id = parts[0]
        c_desc = parts[1] if len(parts) > 1 else constraint
        table_data.append([c_id, c_desc, '✓ Applied'])

    table = ax.table(
        cellText=table_data,
        colLabels=['Constraint ID', 'Description', 'Status'],
        loc='center',
        cellLoc='left',
        colWidths=[0.15, 0.65, 0.2]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

    ax.set_title('CSP Constraints', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()

    return fig


def draw_csp_assignment(csp_result, hospitals, city=None):
    """
    Draw the final CSP assignment showing which hospital was selected
    and why others were rejected.

    Args:
        csp_result (dict): Result from hospital_csp().
        hospitals (list[Hospital]): All hospitals.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(hospitals) + 2)))

    assigned = csp_result['assigned_hospital']
    final_domain = csp_result['domain_reductions'][-1]['domain'] if csp_result['domain_reductions'] else []

    route_costs = csp_result.get('route_costs', {})

    def rejection_reason(name):
        for dr in csp_result['domain_reductions']:
            if name in dr['removed']:
                return dr['reason']
        return 'Rejected'

    # Create table data
    table_data = []
    for h in hospitals:
        status = ''
        color = '#ffffff'
        if assigned and h.name == assigned.name:
            status = '✓ ASSIGNED'
            color = '#2ecc71'
        elif h.name in final_domain:
            status = 'Valid (not selected)'
            color = '#f39c12'
        else:
            status = '✗ ' + rejection_reason(h.name)
            color = '#e74c3c'

        route_cost = route_costs.get(h.name)
        table_data.append([
            h.name,
            (f"Node {h.node}" if city is None or not city.is_real_map()
             else city.label(h.node)[:28]),
            str(h.available_beds),
            'Yes' if h.icu_available else 'No',
            str(h.doctors_available),
            f"{route_cost:.2f}" if route_cost is not None else '-',
            status,
            color
        ])

    cell_text = [row[:7] for row in table_data]
    cell_colors_list = [row[7] for row in table_data]

    table = ax.table(
        cellText=cell_text,
        colLabels=['Hospital', 'Location', 'Avail. Beds', 'ICU', 'Doctors',
                   'Route Cost', 'Status'],
        loc='center',
        cellLoc='center',
        colWidths=[0.22, 0.09, 0.1, 0.07, 0.09, 0.1, 0.27]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for j in range(7):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows
    for i in range(1, len(table_data) + 1):
        status_color = cell_colors_list[i - 1]
        for j in range(7):
            cell = table[i, j]
            if j == 6:  # Status column
                cell.set_facecolor(status_color)
                cell.set_text_props(color='white' if status_color != '#f39c12' else '#2c3e50',
                                    fontweight='bold')
            else:
                cell.set_facecolor('#ffffff')

    title = 'CSP Final Assignment'
    if assigned:
        title += f': {assigned.name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()

    return fig
