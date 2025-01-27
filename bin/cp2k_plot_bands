#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os


def parse_project_bs(filename):
    special_points = {}
    k_points = []
    band_energies = []
    occupations = []
    num_bands = 0
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Process header to extract the number of bands and special points
        for line in lines:
            if line.startswith("# Set"):
                # Extract number of bands from header line
                num_bands = int(line.split()[-2])
            elif line.startswith("#  Special point"):
                # Parse special points
                parts = line.split()
                label = parts[-1].strip()
                # Replace "g", "gamma", or "gam" with the LaTeX symbol for Γ
                if label.lower() in {"g", "gamma", "gam"}:
                    label = r"$\Gamma$"
                try:
                    kx, ky, kz = map(float, parts[4:7])
                    special_points[label] = np.array([kx, ky, kz])
                except ValueError:
                    print(f"Skipping line due to parsing error: {line.strip()}")
            elif line.startswith("#  Point"):
                # Found the start of the k-point data; stop processing header
                break
        
        # Process each k-point and corresponding band energies and occupations
        current_k = None
        energies = []
        occs = []
        
        for line in lines:
            if line.startswith("#  Point"):
                if current_k is not None:
                    k_points.append(current_k)
                    band_energies.append(energies)
                    occupations.append(occs)
                parts = line.split()
                current_k = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                energies = []
                occs = []
            elif re.match(r"\s+\d+", line):
                # Parse band energy and occupation
                parts = line.split()
                energy = float(parts[1])
                occupation = float(parts[2])
                energies.append(energy)
                occs.append(occupation)
        
        # Append the last k-point's data
        if current_k is not None:
            k_points.append(current_k)
            band_energies.append(energies)
            occupations.append(occs)

    # Convert lists to numpy arrays
    k_points = np.array(k_points)
    band_energies = np.array(band_energies)
    occupations = np.array(occupations)

    return k_points, band_energies, occupations, special_points, num_bands

def calculate_k_distances(k_points):
    distances = np.linalg.norm(np.diff(k_points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances

def calculate_band_gap(band_energies, occupations):
    # Identify the valence and conduction bands based on occupations
    occupied_band_indices = np.any(occupations > 0.0, axis=0)
    valence_band_max = np.max(band_energies[:, occupied_band_indices], axis=1)
    conduction_band_min = np.min(band_energies[:, ~occupied_band_indices], axis=1)
    
    # Find the global maximum of the valence band and minimum of the conduction band
    vbm = np.max(valence_band_max)  # Valence band maximum
    cbm = np.min(conduction_band_min)  # Conduction band minimum
    
    # Find the k-point indices where VBM and CBM occur
    vbm_k_index = np.unravel_index(np.argmax(valence_band_max), valence_band_max.shape)
    cbm_k_index = np.unravel_index(np.argmin(conduction_band_min), conduction_band_min.shape)
    
    # Calculate band gap and determine if it's direct or indirect
    band_gap = cbm - vbm
    is_direct = vbm_k_index == cbm_k_index
    
    return band_gap, is_direct

def main():
    parser = argparse.ArgumentParser(description="Process CP2K band structure from project.bs.")
    parser.add_argument("bs_file", help="Path to the project.bs file")
    parser.add_argument("--figsize", type=float, nargs=2, default=[20, 15], help="Figure size in cm (width, height)")
    parser.add_argument("--dpi", type=int, default=100, help="Dots per inch (DPI) for the figure")
    parser.add_argument("--energy_range", type=float, nargs=2, help="Energy range in eV to plot")
    args = parser.parse_args()
    
    # Define output filename based on the bs_file base name
    base_name = os.path.splitext(os.path.basename(args.bs_file))[0]
    output_filename = f"{base_name}.bs.png"
    
    # Convert figsize from cm to inches for Matplotlib
    figsize_inches = (args.figsize[0] / 2.54, args.figsize[1] / 2.54)
    
    # Parse project.bs file
    k_points, band_energies, occupations, special_points, num_bands = parse_project_bs(args.bs_file)
    
    # Calculate cumulative distances between k-points
    k_distances = calculate_k_distances(k_points)
    
    # Determine Fermi level as the maximum energy of the occupied bands
    occupied_band_indices = np.any(occupations > 0.0, axis=0)
    fermi_level = np.max(band_energies[:, occupied_band_indices])
    aligned_energies = band_energies - fermi_level
    
    # Calculate band gap
    band_gap, is_direct = calculate_band_gap(aligned_energies, occupations)
    gap_type = "Direct" if is_direct else "Indirect"
    
    # Determine energy range for plotting
    energy_min, energy_max = args.energy_range if args.energy_range else (aligned_energies.min(), aligned_energies.max())
    
    # Plot setup
    plt.figure(figsize=figsize_inches, dpi=args.dpi)
    for band in range(aligned_energies.shape[1]):
        plt.plot(k_distances, aligned_energies[:, band], color="black", lw=1)
    
    # Configure x-axis for special BZ points with uppercase labels, keeping Gamma as \Gamma
    bz_positions = {
        (label if label == r"$\Gamma$" else label.upper()): [
            k_distances[i] for i, kp in enumerate(k_points) if np.allclose(kp, coords)
        ]
        for label, coords in special_points.items()
    }
    
    # Plot each special point and add vertical lines at each occurrence
    for label, positions in bz_positions.items():
        for pos in positions:
            plt.axvline(pos, color="black", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())
    
    # Set special points as x-axis ticks
    flat_positions = [pos for positions in bz_positions.values() for pos in positions]
    flat_labels = [label for label, positions in bz_positions.items() for _ in positions]
    plt.xticks(flat_positions, flat_labels)
    
    # Set plot limits to remove padding at the beginning and end
    plt.xlim(k_distances.min(), k_distances.max())
    plt.ylim(energy_min, energy_max)
    
    # Y-axis label for energy levels relative to the Fermi level
    plt.ylabel(r"$\epsilon - \epsilon_F$ [eV]")
    plt.axhline(0, color="red", linestyle="--", linewidth=plt.gca().spines["bottom"].get_linewidth())  # Horizontal line at Fermi level
    
    # Save and display the plot
    plt.savefig(output_filename, format="png", bbox_inches="tight")
    plt.show()
    
    # Output band structure information
    print("\n--- Band Structure Information ---")
    print(f"Total number of bands: {num_bands}")
    print(f"Number of valence bands: {np.sum(occupied_band_indices)}")
    print(f"Number of conduction bands: {num_bands - np.sum(occupied_band_indices)}")
    print(f"Band Gap: {band_gap:.2f} eV ({gap_type})")

if __name__ == "__main__":
    main()

