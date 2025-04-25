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


def parse_bs_file(filename):
    """
    Parses a CP2K band structure file to extract k-points, weights, band energies, occupations, special points, and the number of bands.
    Args:
        filename (str): The path to the CP2K band structure file.
    Returns:
        tuple: A tuple containing the following elements:
            - k_points (np.ndarray): An array of k-points with shape (n_kps, 3).
            - weights (np.ndarray): An array of weights corresponding to each k-point with shape (n_kps,).
            - band_energies ((np.ndarray),[(np.ndarray)]): A tuple containing 1 or 2 2D array(s) of band energies with shape (n_kps, n_bands).
            - occupations ((np.ndarray),[(np.ndarray)]): A tuple containing 1 or 2 2D array(s) of occupations with shape (n_kps, n_bands).
            - special_points (dict): A dictionary of special points with their labels as keys and k-point coordinates as values.
            - n_bands (int): The total number of bands. This is the same for both spins in case of spin-polarized calculations.
            - n_spin (int): The number of spins (1 for non-spin-polarized calculations, 2 for spin-polarized calculations).
    """

    special_points = {}
    k_points = []
    weights  = []
    # these have a leading dimension of 2 in case of spin-polarized calculations
    band_energies = [[],[]]
    occupations = [[],[]]

    with open(filename, 'r') as file:
        lines = file.readlines()

        # Process each k-point and corresponding band energies and occupations
        current_energies = []
        sps_counter = 0

        # Process header to extract the number of bands and special points and the k-points
        for line in lines:
            if line.startswith("# Set"):
                # Extract number of bands from header line
                n_sps   = int(line.split()[3])
                n_kps   = int(line.split()[-4])
                # this will be the same for both spins in case of spin-polarized calculations
                # by design of CP2K and how it prints .bs files
                n_bands = int(line.split()[-2])

            elif line.startswith("#  Special point"):
                # Parse special points
                sps_counter += 1
                parts = line.split()
                label = parts[-1].strip()
                # Replace "g", "gamma", or "gam" with the LaTeX symbol for Γ
                if label.lower() in {"g", "gamma", "gam"}:
                    label = r"$\Gamma$"

                # Parse the coordinates of the special point
                try:
                    kx, ky, kz = map(float, parts[4:7])
                    # check if the special point is already in the dictionary
                    if label not in special_points:
                        special_points[label] = np.array([kx, ky, kz])
                    else:
                        # check it is really the same and not just the same label
                        assert(np.allclose(special_points[label], np.array([kx, ky, kz])))
                except ValueError:
                    print(f"Failed in parsing special k-point at line:\n{line.strip()}")

            elif line.startswith("#  Point"):
                # store the energies of the previous k-point
                if len(current_energies) == n_bands:
                    ispin = current_spin - 1
                    if ispin == 0:
                        k_points.append(current_k)
                        weights.append(current_weight)
                    band_energies[ispin].append(current_energies)
                    occupations[ispin].append(current_occs)
                # get the information of the current k-point
                parts = line.split()
                current_spin = int(parts[4][0])
                current_k = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                current_weight = float(parts[8])
                current_energies = []
                current_occs = []

            elif re.match(r"\s+\d+", line):
                # Parse band energy and occupation
                parts = line.split()
                current_energies.append(float(parts[1]))
                current_occs.append(float(parts[2]))

        # Append the last k-point's data
        if len(current_energies) == n_bands:
            ispin = current_spin - 1
            if ispin == 0:
                k_points.append(current_k)
                weights.append(current_weight)
            band_energies[ispin].append(current_energies)
            occupations[ispin].append(current_occs)

    # some sanity checks
    assert(sps_counter == n_sps)
    assert(len(k_points) == n_kps)

    # Convert lists to numpy arrays
    k_points = np.array(k_points)
    weights = np.array(weights)
    n_spins = 1
    if len(band_energies[1]) > 0:
        n_spins = 2
        band_energies = (np.array(band_energies[0]), np.array(band_energies[1]))
        occupations = (np.array(occupations[0]), np.array(occupations[1]))
    else:
        band_energies = (np.array(band_energies[0]),)
        occupations = (np.array(occupations[0]),)

    return k_points, weights, band_energies, occupations, special_points, n_bands, n_spins



def parse_dos_file(filename):
    """ Parses a CP2K DOS file and converts energy from Hartree to eV. """
    if not filename.endswith(".dos"):
        print("Error: The file is not a .dos file.")
        sys.exit(1)

    data = np.loadtxt(filename, comments='#')

    energy = data[:, 0] * HARTREE_TO_EV  # Convert Hartree to eV
    density = [data[:, 1]]
    population = [data[:, 2]]
    if data.shape[1] == 5:
        density.append(data[:, 3])
        population.append(data[:, 4])

    return energy, density, population


def calculate_k_distances(k_points):
    distances = np.linalg.norm(np.diff(k_points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances


def get_fermi_energy(band_energies, occupations):
    """ Determines the Fermi energy as the highest occupied band energy. """
    E_fermi = -math.inf
    for ispin in range(len(band_energies)):
        for k in range(len(band_energies[ispin])):
            occ_bands = occupations[ispin][k,:] > 0.0
            E_fermi = max(E_fermi, np.max(band_energies[ispin][k, occ_bands]))

    return E_fermi


# sigma is in eV
def apply_gaussian_smoothing(energy, density, sigma):
    """ Applies Gaussian smoothing to the DOS data. """
    return gaussian_filter1d(density, sigma=sigma/(energy[1] - energy[0]))


# band_energies and occupations are 2D arrays with shape (n_kps, n_bands)
def calculate_band_gap(band_energies, occupations):
    # Identify the valence and conduction bands based on occupations
    # This assumes uniform occupation among k-points
    occ_bands = np.any(occupations > 0.0, axis=0)
    valence_band_max = np.max(band_energies[:, occ_bands], axis=1)
    conduction_band_min = np.min(band_energies[:, ~occ_bands], axis=1)

    # Find the global maximum of the valence band and minimum of the conduction band
    vbm = np.max(valence_band_max)  # Valence band maximum
    cbm = np.min(conduction_band_min)  # Conduction band minimum

    # Find the k-point indices where VBM and CBM occur
    vbm_k_index = np.unravel_index(np.argmax(valence_band_max), valence_band_max.shape)
    cbm_k_index = np.unravel_index(np.argmin(conduction_band_min), conduction_band_min.shape)

    # Calculate band gap and determine if it's direct or indirect
    band_gap = cbm - vbm

    return band_gap, vbm_k_index, cbm_k_index, np.sum(occ_bands), np.sum(~occ_bands)


def plot_bands(bs_data, dos_data=None, figsize=(10, 6), dpi=150, ewin=None, sigma=0.05, fermi_energy=None):
    """Plots band structure, and optionally adds a DOS plot on the right if dos_data is provided."""

    k_points, _, band_energies, occupations, special_points, n_bands, n_spins = bs_data

    # Get the Fermi energy and align band energies to it
    if fermi_energy is None:
        E_fermi = get_fermi_energy(band_energies, occupations)
    else:
        E_fermi = fermi_energy

    k_distances = calculate_k_distances(k_points)
    # TODO: maybe it's just better to have bands energies in a list, which I can modify
    aligned_energies = tuple(map(lambda x: x - E_fermi, band_energies))

    if dos_data:
        dos_energy, density, _ = dos_data
        dos_energy -= E_fermi  # Align DOS energy to Fermi level

    # find spoecial points positions in k_distances
    bz_positions = {
        (label if label == r"$\Gamma$" else label.upper()): [
            k_distances[i] for i, kp in enumerate(k_points) if np.allclose(kp, coords)
        ]
        for label, coords in special_points.items()
    }

    # needed when plotting both spins
    bs_colors = ["blue", "red"]
    E_vbm = []
    E_cbm = []
    k_vbm = []
    k_cbm = []

    for ispin in range(n_spins):
        band_gap, vbm_k_idx, cbm_k_idx, n_val, n_con = calculate_band_gap(aligned_energies[ispin], occupations[ispin])
        assert(n_val + n_con == n_bands)
        k_vbm.append(vbm_k_idx)
        k_cbm.append(cbm_k_idx)

        gap_type = "Direct" if vbm_k_idx == cbm_k_idx else "Indirect"
        print(f"\n--- Band Structure Information for Spin {ispin+1} ---")
        print(f"Total number of bands: {n_bands}")
        print(f"Number of valence bands: {n_val}")
        print(f"Number of conduction bands: {n_con}")
        print(f"Fermi Energy: {E_fermi:.2f} eV")
        print(f"Band Gap: {band_gap:.2f} eV ({gap_type})")

        fig_width, fig_height = figsize

        if dos_data:
            fig, (ax_band, ax_dos) = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(fig_width, fig_height), dpi=dpi
            )
            fig.subplots_adjust(wspace=0)
            if n_spins == 2 and ispin == 0:
                fig_tot, (ax_band_tot, ax_dos_tot) = plt.subplots(
                    1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(fig_width, fig_height), dpi=dpi
                )
                fig_tot.subplots_adjust(wspace=0)
        else:
            # band structure plot for each spin
            fig, ax_band = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            ax_dos = None
            # band structure plot for both spins combined
            if n_spins == 2 and ispin == 0:
                fig_tot, ax_band_tot = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                ax_dos_tot = None

        # Set title
        ax_band.set_title(f"Band Structure (Spin {ispin+1})")

        # Plot each special point and add vertical lines at each occurrence
        for label, positions in bz_positions.items():
            for pos in positions:
                ax_band.axvline(pos, color="gray", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())
                if ispin == 1:
                    ax_band_tot.axvline(pos, color="gray", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())

        # Set special points as x-axis ticks
        flat_positions = [pos for positions in bz_positions.values() for pos in positions]
        flat_labels = [label for label, positions in bz_positions.items() for _ in positions]
        ax_band.set_xticks(flat_positions)
        ax_band.set_xticklabels(flat_labels)

        ax_band.set_ylabel(r"$E - E_\text{F}$ [eV]")
        # Draw a line at the Fermi energy
        ax_band.axhline(0, color="black", linestyle="--", linewidth=0.6)
        # Set plot limits to remove padding at the beginning and end
        ax_band.set_xlim(k_distances.min(), k_distances.max())

        if ispin == 1:
            ax_band_tot.set_xticks(flat_positions)
            ax_band_tot.set_xticklabels(flat_labels)

            ax_band_tot.set_ylabel(r"$E - E_\text{F}$ [eV]")
            # Draw a line at the Fermi energy
            ax_band_tot.axhline(0, color="black", linestyle="--", linewidth=0.6)
            # Set plot limits to remove padding at the beginning and end
            ax_band_tot.set_xlim(k_distances.min(), k_distances.max())

        # Actually plot band structure
        for band in range(aligned_energies[ispin].shape[1]):
            # for each spin separately
            ax_band.plot(k_distances, aligned_energies[ispin][:, band], color="black", lw=1.5)
            # for both spins combined
            if n_spins == 2:
                ax_band_tot.plot(k_distances, aligned_energies[ispin][:, band], color=bs_colors[ispin], lw=1.5)

        # Add circles at the VBM and CBM positions
        vbm_k_dist = k_distances[k_vbm[ispin]]
        cbm_k_dist = k_distances[k_cbm[ispin]]
        E_vbm.append(aligned_energies[ispin][k_vbm[ispin], n_val - 1])
        E_cbm.append(aligned_energies[ispin][k_cbm[ispin], n_val])
        ax_band.scatter(vbm_k_dist, E_vbm[ispin], color="green", s=20, zorder=5)
        ax_band.scatter(cbm_k_dist, E_cbm[ispin], color="green", s=20, zorder=5)
        if ispin == 1:
            if E_vbm[0] > E_vbm[1]:
                E_vbm_max = E_vbm[0]
                vbm_k_dist_max = k_distances[k_vbm[0]]
            else:
                E_vbm_max = E_vbm[1]
                vbm_k_dist_max = k_distances[k_vbm[1]]

            if E_cbm[0] < E_cbm[1]:
                E_cbm_min = E_cbm[0]
                cbm_k_dist_min = k_distances[k_cbm[0]]
            else:
                E_cbm_min = E_cbm[1]
                cbm_k_dist_min = k_distances[k_cbm[1]]

            ax_band_tot.scatter(vbm_k_dist_max, E_vbm_max, color="green", s=20, zorder=5)
            ax_band_tot.scatter(cbm_k_dist_min, E_cbm_min, color="green", s=20, zorder=5)
            ax_band_tot.set_title("Band Structure (Spin 1 and 2)")

        # set limits if requested
        if ewin:
            ax_band.set_ylim(ewin)
            if ispin == 1:
                ax_band_tot.set_ylim(ewin)

        # plot DOS
        if ax_dos:
            density[ispin] = apply_gaussian_smoothing(dos_energy, density[ispin], sigma)

            ax_dos.fill_betweenx(dos_energy, 0, density[ispin], color="lightgray", alpha=0.8)
            ax_dos.plot(density[ispin], dos_energy, color="gray", lw=1.5)

            ax_dos.axvline(0, color="black", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())
            ax_dos.set_yticklabels([])
            ax_dos.set_xticks([])
            ax_dos.spines["left"].set_visible(False)

            # get the highest value of the density within the visible energy range
            max_density = np.max(density[ispin][(dos_energy >= ax_band.get_ylim()[0]) & (dos_energy <= ax_band.get_ylim()[1])])
            ax_dos.set_xlim(0, max_density * 1.1)
            ax_dos.set_ylim(ax_band.get_ylim())
            # we need to do the total plot here
            if ispin == 1:
                ax_dos_tot.fill_betweenx(dos_energy, 0, sum(density), color="lightgray", alpha=0.8)
                ax_dos_tot.plot(sum(density), dos_energy, color="gray", lw=1.5)

                ax_dos_tot.axvline(0, color="black", linestyle="-", linewidth=plt.gca().spines["bottom"].get_linewidth())
                ax_dos_tot.set_yticklabels([])
                ax_dos_tot.set_xticks([])
                ax_dos_tot.spines["left"].set_visible(False)

                # get the highest value of the density within the visible energy range
                max_density = np.max(sum(density)[(dos_energy >= ax_band_tot.get_ylim()[0]) & (dos_energy <= ax_band_tot.get_ylim()[1])])
                ax_dos_tot.set_xlim(0, max_density * 1.1)
                ax_dos_tot.set_ylim(ax_band_tot.get_ylim())

    plt.show()

# TODO: fix for spin-polarized calculations
def plot_tdos(dos_data, figsize=(10, 6), dpi=150, sigma=0.05, ewin=None, fermi_energy=None):
    """Plots total DOS separately with energy on the x-axis."""

    dos_energy, density, population = dos_data

    # Find the largest energy that has population different from zero
    # Determine Fermi energy
    if fermi_energy is None:
        non_zero_population_indices = np.where(sum(population) != 0)[0]
        if len(non_zero_population_indices) > 0:
            E_fermi = dos_energy[non_zero_population_indices[-1]]
            print(f"Fermi Energy: {E_fermi:.2f} eV")
        else:
            print("No non-zero population found in the DOS data.")
            E_fermi = 0.0 # set to zero so that we don't crash
    else:
        E_fermi = fermi_energy
        print(f"User-provided Fermi Energy: {E_fermi:.2f} eV")

    dos_energy -= E_fermi  # Align DOS energy to Fermi level
    for ispin in range(len(density)):
        density[ispin] = apply_gaussian_smoothing(dos_energy, density[ispin], sigma)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.fill_between(dos_energy, 0, sum(density), color="lightgray", alpha=0.8)
    ax.plot(dos_energy, sum(density), color="gray", lw=1.5)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density of States")
    ax.set_title("Total Density of States (TDOS)")
    if ewin:
        ax.set_xlim(ewin)

    # max_density = np.max(sum(density)[(dos_energy >= ewin[0]) & (dos_energy <= ewin[1])])
    max_density = np.max(sum(density)[(dos_energy >= ax.get_xlim()[0]) & (dos_energy <= ax.get_xlim()[1])])
    ax.set_ylim(0, max_density * 1.1)

    # set the grid only vertically
    ax.xaxis.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze CP2K band structure and DOS.")
    parser.add_argument("--bs", type=str, help="Path to the CP2K band structure file (project.bs)")
    parser.add_argument("--dos", type=str, help="Path to the CP2K DOS file (project.dos)")
    parser.add_argument("--ewin", type=float, nargs=2, default=None, help="Energy window for plots (eV)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6], help="Figure size in cm (width, height). Default is 10x6 cm.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution in DPI. Default is 150.")
    parser.add_argument("--sigma", type=float, default=0.05, help="Gaussian broadening parameter in eV. Default is 0.05.")
    parser.add_argument("--fermi", type=float, default=None, help="Fixed Fermi energy (in eV). If not provided, it will be calculated from the band structure.")

    args = parser.parse_args()

    bs_data = parse_bs_file(args.bs) if args.bs else None
    dos_data = parse_dos_file(args.dos) if args.dos else None

    if bs_data:
        plot_bands(bs_data, dos_data, args.figsize, args.dpi, args.ewin, args.sigma, fermi_energy=args.fermi)
    elif dos_data:
        plot_tdos(dos_data, args.figsize, args.dpi, args.sigma, args.ewin, fermi_energy=args.fermi)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
