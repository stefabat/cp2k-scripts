#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
from matplotlib.lines import Line2D
from itertools import cycle

def parse_angular_momentum(column_name):
    """
    Extract the angular momentum type from the column name.
    For example:
        's'   -> 's'
        'py'  -> 'p'
        'd-2' -> 'd'
        'g0'  -> 'g'
    """
    match = re.match(r'([spdfg])', column_name.lower())
    if match:
        return match.group(1)
    else:
        return 'other'

def get_color_map(angular_momenta):
    """
    Assign distinct colors to each angular momentum type present.
    """
    # Define a list of distinct colors
    predefined_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 'magenta', 'brown']
    
    if len(angular_momenta) > len(predefined_colors):
        print("Warning: More angular momenta than predefined colors. Some colors will be reused.")
    
    color_cycle = cycle(predefined_colors)
    return {ang: next(color_cycle) for ang in angular_momenta}

def get_line_styles():
    """
    Define a list of distinct line styles.
    """
    return ['-', '--', '-.', ':']

def plot_pdos(csv_file):
    # Check if file exists
    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)

    # Read the space-separated file with raw string for regex
    try:
        data = pd.read_csv(csv_file, sep=r'\s+')
    except Exception as e:
        print(f"Error reading '{csv_file}': {e}")
        sys.exit(1)

    # Check if there are at least two columns
    if data.shape[1] < 2:
        print("Error: File must contain at least two columns (energy and PDOS components).")
        sys.exit(1)

    # Assume first column is energy
    energy = data.iloc[:, 0]
    pdos_data = data.iloc[:, 1:]
    headers = pdos_data.columns.tolist()

    # Parse angular momentum types
    ang_mom = {}
    for col in headers:
        ang = parse_angular_momentum(col)
        ang_mom[col] = ang

    # Identify unique angular momenta present (excluding 'other')
    unique_ang_mom = sorted(set(ang for ang in ang_mom.values() if ang != 'other'))

    if not unique_ang_mom:
        print("Error: No valid angular momentum types found in the data.")
        sys.exit(1)

    # Assign colors to angular momenta
    color_map = get_color_map(unique_ang_mom)

    # Assign line styles to components within each angular momentum
    line_styles = get_line_styles()
    line_style_cycle = cycle(line_styles)

    # Initialize a dictionary to hold total PDOS per angular momentum
    total_pdos = {ang: pd.Series(0, index=energy.index) for ang in unique_ang_mom}

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # -------------------------
    # First Subplot: Individual PDOS Components
    # -------------------------
    ax1 = axes[0]
    legend_elements = []
    # Dictionary to keep track of assigned line styles per angular momentum
    ang_line_styles = {ang: cycle(line_styles) for ang in unique_ang_mom}

    for col in headers:
        ang = ang_mom[col]
        if ang == 'other':
            continue  # Skip 'other' or handle differently if desired
        color = color_map.get(ang, 'gray')  # Default to 'gray' if not found
        linestyle = next(ang_line_styles[ang])
        ax1.plot(energy, pdos_data[col], label=col, color=color, linestyle=linestyle, linewidth=1.5)
        # Sum PDOS for total plot
        total_pdos[ang] += pdos_data[col]

    # Create custom legend for individual components
    for ang in unique_ang_mom:
        # Create a sample line for each angular momentum
        linestyle_samples = get_line_styles()
        for ls in linestyle_samples:
            legend_elements.append(Line2D([0], [0], color=color_map[ang], lw=2, linestyle=ls, label=f"{ang.upper()} - {ls}"))
    
    # To avoid duplicate labels in legend, create unique legend entries
    # Alternatively, since line styles represent components, you might want to map them to actual components
    # This requires knowing which line style corresponds to which component, which isn't straightforward
    # Instead, it's better to create a legend that indicates line styles represent different components
    # and colors represent angular momenta

    # Here's an improved legend approach:
    # 1. Create legend entries for angular momenta with colors
    # 2. Create separate legend entries for line styles representing components

    # Create legend for angular momenta colors
    ang_legend_elements = [Line2D([0], [0], color=color_map[ang], lw=2, label=ang.upper()) for ang in unique_ang_mom]
    # Create legend for line styles
    comp_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=ls, label=ls) for ls in line_styles]

    # Add both legends to the plot
    legend1 = ax1.legend(handles=ang_legend_elements, title='Angular Momentum', loc='upper right')
    legend2 = ax1.legend(handles=comp_legend_elements, title='Component Line Styles', loc='upper left')
    ax1.add_artist(legend1)  # Add the first legend back

    ax1.set_xlabel('Energy (eV)', fontsize=14)
    ax1.set_ylabel('PDOS', fontsize=14)
    ax1.set_title('Projected Density of States (Individual Components)', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # -------------------------
    # Second Subplot: Total PDOS per Angular Momentum
    # -------------------------
    ax2 = axes[1]
    total_legend_elements = []
    for ang in unique_ang_mom:
        pdos = total_pdos[ang]
        color = color_map[ang]
        ax2.plot(energy, pdos, label=ang.upper(), color=color, linewidth=2)
        total_legend_elements.append(Line2D([0], [0], color=color, lw=2, label=ang.upper()))

    ax2.set_xlabel('Energy (eV)', fontsize=14)
    ax2.set_ylabel('Total PDOS', fontsize=14)
    ax2.set_title('Projected Density of States (Total per Angular Momentum)', fontsize=16)
    ax2.legend(handles=total_legend_elements, title='Angular Momentum', fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: ./plot_pdos.py <path_to_space_separated_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    plot_pdos(csv_file)

if __name__ == "__main__":
    main()

