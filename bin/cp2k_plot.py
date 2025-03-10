#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import re
import math
import os
from scipy.ndimage import gaussian_filter1d

# Ensure Matplotlib's save dialog opens in the current working directory
plt.rcParams["savefig.directory"] = os.getcwd()

HARTREE_TO_EV = 27.211384  # Conversion factor from Hartree to eV


def parse_bs_file(filename):
    """
    Parses a CP2K band structure file to extract k-points, weights, band energies, occupations, special points, and the number of bands.
    Args:
        filename (str): The path to the CP2K band structure file.
    Returns:
        tuple: A tuple containing the following elements:
            - k_points (np.ndarray): An array of k-points with shape (num_k_points, 3).
            - weights (np.ndarray): An array of weights corresponding to each k-point with shape (num_k_points,).
            - band_energies (np.ndarray): A 2D array of band energies for each k-point with shape (num_k_points, num_bands).
            - occupations (np.ndarray): A 2D array of occupations for each k-point with shape (num_k_points, num_bands).
            - special_points (dict): A dictionary of special points with their labels as keys and k-point coordinates as values.
            - num_bands (int): The number of bands.

        All the arrays have a leading dimension of 2 in case of spin-polarized calculations, e.g. k_points has a shape of (2, num_k_points, 3).
    """

    special_points = {}
    k_points = [[],[]]
    weights  = [[],[]]
    band_energies = [[],[]]
    occupations = [[],[]]

    with open(filename, 'r') as file:
        lines = file.readlines()

        # Process each k-point and corresponding band energies and occupations
        energies = []

        # Process header to extract the number of bands and special points and the k-points
        for line in lines:
            if line.startswith("# Set"):
                # Extract number of bands from header line
                num_sp    = int(line.split()[3])
                num_kps   = int(line.split()[-4])
                num_bands = int(line.split()[-2])
                # print(f"Number of special points: {num_sp}")
                # print(f"Number of k-points: {num_kps}")
                # print(f"Number of bands: {num_bands}")
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
                # store the energies of the previous k-point
                if len(energies) == num_bands:
                    ispin = current_spin - 1
                    k_points[ispin].append(current_k)
                    weights[ispin].append(current_weight)
                    band_energies[ispin].append(energies)
                    occupations[ispin].append(occs)
                # get the information of the current k-point
                parts = line.split()
                current_spin = int(parts[4][0])
                current_k = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                current_weight = float(parts[8])
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
        if len(energies) == num_bands:
            ispin = current_spin - 1
            k_points[ispin].append(current_k)
            weights[ispin].append(current_weight)
            band_energies[ispin].append(energies)
            occupations[ispin].append(occs)

    # some sanity checks
    # assert(len(special_points) == num_sp)
    assert(len(k_points[0]) == num_kps)

    # Convert lists to numpy arrays
    # Assume that the number of k-points is the same for both spins in case of spin-polarized calculations
    num_spin = 1
    if len(k_points[0]) == len(k_points[1]):
        k_points = np.array(k_points)
        weights = np.array(weights)
        band_energies = np.array(band_energies)
        occupations = np.array(occupations)
        num_spin = 2
    elif len(k_points[1]) == 0:
        k_points = np.array(k_points[0])
        weights = np.array(weights[0])
        band_energies = np.array(band_energies[0])
        occupations = np.array(occupations[0])
    else:
        print("Error: The number of k-points is different for the two spins.")
        sys.exit(1)

    return k_points, weights, band_energies, occupations, special_points, num_bands, num_spin



def parse_dos_file(filename):
    """ Parses a CP2K DOS file and converts energy from Hartree to eV. """
    if not filename.endswith(".dos"):
        print("Error: The file is not a .dos file.")
        sys.exit(1)

    data = np.loadtxt(filename, comments='#')

    energy = data[:, 0] * HARTREE_TO_EV  # Convert Hartree to eV
    density = data[:, 1]
    population = data[:, 2]

    return np.array(energy), np.array(density), np.array(population)


def calculate_k_distances(k_points):
    distances = np.linalg.norm(np.diff(k_points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances


def get_fermi_energy(band_energies, occupations):
    """ Determines the Fermi energy. """
    occupied_band_indices = np.any(occupations > 0.0, axis=0)
    return np.max(band_energies[:, occupied_band_indices])


def apply_gaussian_smoothing(energy, density, sigma_hartree):
    """ Applies Gaussian smoothing to the DOS data. """
    sigma_ev = sigma_hartree * HARTREE_TO_EV  # Convert sigma to eV
    return gaussian_filter1d(density, sigma=sigma_ev / (energy[1] - energy[0]))


# band_energies and occupations are 2D arrays with shape (n_k_points, n_bands)
def calculate_band_gap(band_energies, occupations):
    # Identify the valence and conduction bands based on occupations
    # This assumes uniform occupation among k-points
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

    return band_gap, vbm_k_index, cbm_k_index, np.sum(occupied_band_indices), np.sum(~occupied_band_indices)


def plot_bands(bs_data, dos_data=None, figsize=(10, 6), dpi=150, ewin=None, sigma=0.005):
    """Plots band structure, and optionally adds a DOS plot on the right if dos_data is provided."""

    k_points, _, band_energies, occupations, special_points, num_bands, num_spin = bs_data

    # Align band energies to the Fermi level of alpha spin
    if num_spin == 2:
        k_distances = calculate_k_distances(k_points[0])
        E_fermi = get_fermi_energy(band_energies[0], occupations[0])
    else:
        k_distances = calculate_k_distances(k_points)
        E_fermi = get_fermi_energy(band_energies, occupations)

    aligned_energies = band_energies - E_fermi

    # Calculate band gap based on alpha
    if num_spin == 2:
        band_gap, vbm_kpt, cbm_kpt, n_val, n_con = calculate_band_gap(aligned_energies[0], occupations[0])
    else:
        band_gap, vbm_kpt, cbm_kpt, n_val, n_con = calculate_band_gap(aligned_energies, occupations)
    gap_type = "Direct" if vbm_kpt == cbm_kpt else "Indirect"
    print("\n--- Band Structure Information ---")
    print(f"Total number of bands: {num_bands}")
    print(f"Number of valence bands: {n_val}")
    print(f"Number of conduction bands: {n_con}")
    print(f"Fermi Energy: {E_fermi:.2f} eV")
    print(f"Band Gap: {band_gap:.2f} eV ({gap_type})")


    fig_width, fig_height = figsize

    if dos_data:
        fig, (ax_band, ax_dos) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(fig_width, fig_height), dpi=dpi
        )
        dos_energy, density, _ = dos_data
        dos_energy -= E_fermi  # Align DOS energy to Fermi level
        density = apply_gaussian_smoothing(dos_energy, density, sigma)
    else:
        fig, ax_band = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax_dos = None

    if num_spin == 2:
        for band in range(aligned_energies[0].shape[1]):
            ax_band.plot(k_distances, aligned_energies[0, :, band], color="red", lw=2)
            ax_band.plot(k_distances, aligned_energies[1, :, band], color="blue", lw=1.5)
    else:
        for band in range(aligned_energies.shape[1]):
            ax_band.plot(k_distances, aligned_energies[:, band], color="black", lw=1.5)

    if num_spin == 2:
        _k_points = k_points[0]
    else:
        _k_points = k_points

    bz_positions = {
        (label if label == r"$\Gamma$" else label.upper()): [
            k_distances[i] for i, kp in enumerate(_k_points) if np.allclose(kp, coords)
        ]
        for label, coords in special_points.items()
    }

    # Plot each special point and add vertical lines at each occurrence
    for label, positions in bz_positions.items():
        for pos in positions:
            ax_band.axvline(pos, color="black", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())

    # Set special points as x-axis ticks
    flat_positions = [pos for positions in bz_positions.values() for pos in positions]
    flat_labels = [label for label, positions in bz_positions.items() for _ in positions]
    ax_band.set_xticks(flat_positions)
    ax_band.set_xticklabels(flat_labels)

    # for label, positions in bz_positions.items():
    #     for pos in positions:
    #         ax_band.axvline(pos, color="black", linestyle="-", linewidth=1)

    ax_band.set_ylabel(r"$\epsilon - \epsilon_F$ [eV]")
    # Draw a line at the Fermi energy
    ax_band.axhline(0, color="red", linestyle="--", linewidth=1)
    # Set plot limits to remove padding at the beginning and end
    ax_band.set_xlim(k_distances.min(), k_distances.max())

    if ewin:
        ax_band.set_ylim(ewin)

    # Add circles at the VBM and CBM positions
    vbm_k_distance = k_distances[vbm_kpt]
    cbm_k_distance = k_distances[cbm_kpt]
    if num_spin == 2:
        vbm_energy = aligned_energies[0, vbm_kpt[0], n_val - 1]
        cbm_energy = aligned_energies[0, cbm_kpt[0], n_val]
    else:
        vbm_energy = aligned_energies[vbm_kpt[0], n_val - 1]
        cbm_energy = aligned_energies[cbm_kpt[0], n_val]

    ax_band.scatter(vbm_k_distance, vbm_energy, color="green", s=100, edgecolor='black', zorder=5, label='VBM')
    ax_band.scatter(cbm_k_distance, cbm_energy, color="purple", s=100, edgecolor='black', zorder=5, label='CBM')

    if ax_dos:
        ax_dos.fill_betweenx(dos_energy, 0, density, color="lightgray", alpha=0.8)
        ax_dos.plot(density, dos_energy, color="gray", lw=1.5)

        ax_dos.axvline(0, color="black", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())
        ax_dos.set_yticklabels([])
        ax_dos.set_xticks([])
        ax_dos.spines["left"].set_visible(False)
        plt.subplots_adjust(wspace=0)

        # get the highest value of the density within the visible energy range
        max_density = np.max(density[(dos_energy >= ax_band.get_ylim()[0]) & (dos_energy <= ax_band.get_ylim()[1])])
        ax_dos.set_xlim(0, max_density * 1.1)
        ax_dos.set_ylim(ax_band.get_ylim())

    plt.show()


def plot_tdos(dos_data, figsize=(10, 6), dpi=150, sigma=0.02, ewin=None):
    """Plots total DOS separately with energy on the x-axis."""

    dos_energy, density, population = dos_data
    # Find the largest energy that has population different from zero
    non_zero_population_indices = np.where(population != 0)[0]
    if len(non_zero_population_indices) > 0:
        E_fermi = dos_energy[non_zero_population_indices[-1]]
        print(f"Fermi Energy: {E_fermi:.2f} eV")
    else:
        print("No non-zero population found in the DOS data.")
    dos_energy -= E_fermi  # Align DOS energy to Fermi level
    density = apply_gaussian_smoothing(dos_energy, density, sigma)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.fill_between(dos_energy, 0, density, color="lightgray", alpha=0.8)
    ax.plot(dos_energy, density, color="gray", lw=1.5)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density of States")
    ax.set_title("Total Density of States (TDOS)")
    if ewin:
        ax.set_xlim(ewin)

    max_density = np.max(density[(dos_energy >= ewin[0]) & (dos_energy <= ewin[1])])
    ax.set_ylim(0, max_density * 1.1)

    # set the grid only vertically
    ax.xaxis.grid(True)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze CP2K band structure and DOS.")
    parser.add_argument("--bs", type=str, help="Path to the CP2K band structure file (project.bs)")
    parser.add_argument("--dos", type=str, help="Path to the CP2K DOS file (project.dos)")
    parser.add_argument("--ewin", type=float, nargs=2, default=None, help="Energy window for plots (eV)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6], help="Figure size in cm (width, height)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution (DPI)")
    parser.add_argument("--sigma", type=float, default=0.005, help="Gaussian broadening parameter in Hartree")

    args = parser.parse_args()

    bs_data = parse_bs_file(args.bs) if args.bs else None
    dos_data = parse_dos_file(args.dos) if args.dos else None

    if bs_data:
        plot_bands(bs_data, dos_data, args.figsize, args.dpi, args.ewin, args.sigma)
    elif dos_data:
        plot_tdos(dos_data, args.figsize, args.dpi, args.sigma, args.ewin)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
