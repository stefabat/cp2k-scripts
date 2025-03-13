import numpy as np
import matplotlib.pyplot as plt
import re

# Gaussian broadening function
def gaussian_broadening(energies, dos, sigma, energy_grid):
    broadened_dos = np.zeros_like(energy_grid)
    for e, d in zip(energies, dos):
        broadened_dos += d * np.exp(-((energy_grid - e) ** 2) / (2 * sigma ** 2))
    broadened_dos /= (sigma * np.sqrt(2 * np.pi))  # Normalize
    return broadened_dos

# --- Input Parameters ---
num_elements = int(input("Enter the number of elements: "))
file_prefix = input("Enter the filename prefix: ")

element_files = {}
for i in range(1, num_elements + 1):
    element_files[i] = f"{file_prefix}_k{i}-1.pdos"

# --- Data Storage ---
element_names = {}
orbital_choices = {}
orbital_colors = {}
element_dos = {}
plot_order = []
sigma = 0.1  # Broadening width in eV
total_dos = np.zeros(0)  # Initialize as empty array
fermi_energy = 0.0  # Initialize Fermi energy

# --- Color Palette ---
available_colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']
color_index = 0

# --- File Processing and Data Extraction ---
for i, file in element_files.items():
    try:
        with open(file, 'r') as f:
            lines = f.readlines()

            # Extract element name
            match = re.search(r"atomic kind (\w+)", lines[0])
            if match:
                element = match.group(1)
                element_names[i] = element
            else:
                print(f"Could not extract element name from {file}. Please check the file format.")
                continue

            # Extract Fermi energy
            fermi_match = re.search(r"E\(Fermi\) =[\s]*([-+]?\d*\.\d+)", lines[0])
            if fermi_match:
                fermi_energy = float(fermi_match.group(1)) * 27.2114  # Hartree to eV
                print(f"Fermi energy from {file}: {fermi_energy} eV")
            else:
                print(f"Could not extract Fermi energy from {file}. Using default value.")

            data = np.loadtxt(lines[2:])
            e_values = data[:, 1] * 27.2114  # Convert Hartree to eV
            e_values -= fermi_energy  # Shift by Fermi energy

            s_orbital = data[:, 3]
            p_orbital = data[:, 4] if data.shape[1] > 4 else np.zeros_like(s_orbital)
            d_orbital = data[:, 5] if data.shape[1] > 5 else np.zeros_like(s_orbital)
            f_orbital = data[:, 6] if data.shape[1] > 6 else np.zeros_like(s_orbital)

            raw_total_dos = s_orbital + p_orbital + d_orbital + f_orbital

            # Ask user for orbital choices
            print(f"Available orbitals for {element}: s, p, d, f")
            orbital_input = input(f"Enter orbitals to plot (comma-separated, e.g., s,p,d) or 'none': ").lower()
            orbital_choices[element] = [o.strip() for o in orbital_input.split(',') if o.strip() in ['s', 'p', 'd', 'f']]

            # Ask user for color for each orbital
            for orbital in orbital_choices[element]:
                key = f"{element}_{orbital}"
                color = input(f"Enter color for {element} ({orbital}) (e.g., blue, red, green) or leave blank for default: ").lower()
                if color == "":
                    orbital_colors[key] = available_colors[color_index % len(available_colors)]  # Default color
                    color_index += 1
                else:
                    orbital_colors[key] = color

                element_dos[key] = (gaussian_broadening(e_values, eval(f"{orbital}_orbital"), sigma, e_values))
                plot_order.append(key)

            if total_dos.size == 0:
                total_dos = gaussian_broadening(e_values, raw_total_dos, sigma, e_values)  # Initialize
            else:
                total_dos += gaussian_broadening(e_values, raw_total_dos, sigma, e_values)

    except FileNotFoundError:
        print(f"Error: File {file} not found.")
    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")

# --- Plotting ---

# Determine the energy grid based on the data
x_min = float(input("Enter the desired minimum x-axis value: "))
x_max = float(input("Enter the desired maximum x-axis value: "))

energy_grid = np.linspace(x_min, x_max, 1000)

# Recompute broadened DOS on the determined energy grid
for key in element_dos:
    element = key.split('_')[0]
    orbital = key.split('_')[1]
    element_dos[key] = np.zeros_like(energy_grid)  # Initialize with zeros

    for i, file in element_files.items():
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                match = re.search(r"atomic kind (\w+)", lines[0])
                el = match.group(1)
                if el == element:
                    data = np.loadtxt(lines[2:])
                    e_values = data[:, 1] * 27.2114 - fermi_energy
                    s_orbital = data[:, 3]
                    p_orbital = data[:, 4] if data.shape[1] > 4 else np.zeros_like(s_orbital)
                    d_orbital = data[:, 5] if data.shape[1] > 5 else np.zeros_like(s_orbital)
                    f_orbital = data[:, 6] if data.shape[1] > 6 else np.zeros_like(s_orbital)

                    orbital_data = eval(f"{orbital}_orbital")
                    element_dos[key] = gaussian_broadening(e_values, orbital_data, sigma, energy_grid)

        except FileNotFoundError:
            print(f"Error: File {file} not found.")
        except Exception as e:
            print(f"An error occurred while processing {file}: {e}")


total_dos = np.zeros_like(energy_grid)  # Reinitialize total_dos

for i, file in element_files.items():  # Re-read data for total DOS
    try:
        with open(file, 'r') as f:
            lines = f.readlines()

            data = np.loadtxt(lines[2:])
            e_values = data[:, 1] * 27.2114  # Convert Hartree to eV
            e_values -= fermi_energy  # Shift by Fermi energy

            s_orbital = data[:, 3]
            p_orbital = data[:, 4] if data.shape[1] > 4 else np.zeros_like(s_orbital)
            d_orbital = data[:, 5] if data.shape[1] > 5 else np.zeros_like(s_orbital)
            f_orbital = data[:, 6] if data.shape[1] > 6 else np.zeros_like(s_orbital)

            raw_total_dos = s_orbital + p_orbital + d_orbital + f_orbital
            total_dos += gaussian_broadening(e_values, raw_total_dos, sigma, energy_grid)  # Total DOS needs to be computed on the new grid

    except FileNotFoundError:
        print(f"Error: File {file} not found.")
    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")

plt.figure(figsize=(8, 6))

for key in plot_order:
    element = key.split('_')[0]
    orbital = key.split('_')[1]
    plt.fill_between(energy_grid, element_dos[key], alpha=0.6, label=f"{element} ({orbital})", color=orbital_colors[key])

# Plot total DOS
plt.plot(energy_grid, total_dos, label="Total DOS", color="black", linewidth=1)

# Formatting the plot
plt.xlabel("E-E$_{\\rm F}$ [eV]", fontsize=16)
plt.ylabel("pDOS [states/eV]", fontsize=16)
plt.xlim(x_min, x_max)
plt.legend(fontsize=12)
plt.grid(True)
plt.title("pDOS at the Γ-point", fontsize=16)
plt.show()
