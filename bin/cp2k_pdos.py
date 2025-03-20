import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

def gaussian_broadening(energies, dos, sigma, energy_grid):
    broadened_dos = np.zeros_like(energy_grid)
    for e, d in zip(energies, dos):
        broadened_dos += d * np.exp(-((energy_grid - e) ** 2) / (2 * sigma ** 2))
    broadened_dos /= (sigma * np.sqrt(2 * np.pi))
    return broadened_dos

def plot_pdos(filename_prefix, num_elements):
    # --- File Setup ---
    element_files = {}
    for i in range(1, num_elements + 1):
        element_files[i] = f"{filename_prefix}_k{i}-1.pdos"

    # --- Data Storage ---
    element_names, orbital_choices, orbital_colors = {}, {}, {}
    element_dos, plot_order = {}, []
    sigma, fermi_energy = 0.1, 0.0
    total_dos = np.zeros(0)
    fermi_energy_printed = False  # Flag to track if Fermi energy has been printed

    # --- Color Palette ---
    available_colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan',
                        'magenta', 'brown', 'pink', 'gray']
    color_index = 0

    # --- Print Available Colors ---
    print("Available colors for orbital selection:")
    print(available_colors)
    print("\nIf you press Enter without typing a color, the default color will be assigned.")

    # --- File Processing ---
    for i, file in element_files.items():
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                match = re.search(r"atomic kind (\w+)", lines[0])
                if not match:
                    print(f"Could not extract element name from {file}.")
                    continue
                element = match.group(1)
                element_names[i] = element

                # Extract Fermi energy
                fermi_match = re.search(r"E\(Fermi\) =[\s]*([-+]?\d*\.\d+)", lines[0])
                if fermi_match:
                    fermi_energy = float(fermi_match.group(1)) * 27.2114  # Hartree to eV
                    if not fermi_energy_printed:
                        print(f"Fermi energy from {file}: {fermi_energy:.3f} eV")
                        fermi_energy_printed = True

                data = np.loadtxt(lines[2:])
                e_values = data[:, 1] * 27.2114 - fermi_energy  # Convert and shift

                # Orbital data extraction
                orbitals = {
                    's': data[:, 3],
                    'p': data[:, 4] if data.shape[1] > 4 else np.zeros_like(data[:, 3]),
                    'd': data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 3]),
                    'f': data[:, 6] if data.shape[1] > 6 else np.zeros_like(data[:, 3])
                }

                # Get user input for orbitals
                print(f"\nElement: {element}")
                orbital_input = input("Enter orbitals to plot (comma-separated, e.g., s,p,d) or 'none': ").lower()
                selected_orbitals = [o.strip() for o in orbital_input.split(',') if o.strip() in orbitals]

                # Handle color selection
                for orbital in selected_orbitals:
                    key = f"{element}_{orbital}"
                    color = input(f"Color for {element} ({orbital}) orbital (enter for default - {available_colors[color_index % len(available_colors)]}): ").lower()

                    # Check if the entered color is valid
                    if color and color not in available_colors:
                        print(f"Invalid color '{color}'. Using default color: {available_colors[color_index % len(available_colors)]}")
                        color = ""  # Reset color to empty string to use default

                    orbital_colors[key] = color or available_colors[color_index % len(available_colors)]
                    color_index += 0 if color else 1  # Only advance index for default colors

                    # Store DOS data
                    element_dos[key] = gaussian_broadening(e_values, orbitals[orbital], sigma, e_values)
                    plot_order.append(key)

                # Update total DOS
                raw_total_dos = sum(orbitals.values())  # Sum ALL orbitals
                if total_dos.size == 0:
                    total_dos = gaussian_broadening(e_values, raw_total_dos, sigma, e_values)
                else:
                    total_dos += gaussian_broadening(e_values, raw_total_dos, sigma, e_values)

        except FileNotFoundError:
            print(f"Error: File {file} not found.")
            continue
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    # --- Energy Range Setup ---
    x_min = float(input("\nEnter minimum energy (eV): "))
    x_max = float(input("\nEnter maximum energy (eV): "))
    energy_grid = np.linspace(x_min, x_max, 1000)

    # --- Final Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot individual components
    for key in plot_order:
        element, orbital = key.split('_')
        dos = gaussian_broadening(e_values, element_dos[key], sigma, energy_grid)
        plt.fill_between(energy_grid, dos, alpha=0.6,
                         label=f"{element} ({orbital})",  # Orbital in parentheses
                         color=orbital_colors[key])

    # Plot total DOS
    total_dos_smoothed = gaussian_broadening(e_values, total_dos, sigma, energy_grid)
    plt.plot(energy_grid, total_dos_smoothed, color='black', linewidth=1, label='Total DOS')

    # Format plot
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)  # Fermi level indicator
    plt.xlabel("E - E$_F$ [eV]", fontsize=16)  # eV in square brackets, fontsize 16
    plt.ylabel("Projected DOS [states/eV]", fontsize=16)  # fontsize 16
    #plt.title(f"Projected DOS for {filename_prefix}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(x_min, x_max)

    # Set tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot CP2K projected density of states (pDOS)")
    parser.add_argument("--filename-prefix", type=str, required=True,
                        help="Prefix for PDOS files (e.g., 'COF-1_Zn_0GPa-ALPHA' for files like COF-1_Zn_0GPa-ALPHA_k1-1.pdos)")
    parser.add_argument("--num-elements", type=int, required=True,
                        help="Number of constituent elements in the system")
    args = parser.parse_args()
    plot_pdos(args.filename_prefix, args.num_elements)

if __name__ == "__main__":
    main()
