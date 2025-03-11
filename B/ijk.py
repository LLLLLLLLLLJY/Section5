#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve

def compute_quantities(energy_levels, beta_values, particle_count):
    chemical_potential_list = []
    ground_state_list = []
    d_ground_state_dT_list = []
    heat_capacity_list = []

    for beta in beta_values:
        chemical_potential = fsolve(
            lambda mu: np.sum(1 / (np.exp(beta * (energy_levels - mu)) - 1)) - particle_count,
            energy_levels[0] - 1 / beta / particle_count
        )[0]

        energy_diff = energy_levels - chemical_potential
        exp_beta_energy = np.exp(beta * energy_diff)
        occupation_numbers = 1 / (exp_beta_energy - 1)
        derivative = exp_beta_energy * occupation_numbers**2
        d_occupation_dT = derivative * beta**2 * (energy_diff - np.sum(derivative * energy_diff) / np.sum(derivative))

        chemical_potential_list.append(chemical_potential)
        ground_state_list.append(occupation_numbers[0])
        d_ground_state_dT_list.append(d_occupation_dT[0])
        heat_capacity_list.append(np.sum(d_occupation_dT * energy_levels))

    return (
        np.array(chemical_potential_list),
        np.array(ground_state_list),
        np.array(d_ground_state_dT_list),
        np.array(heat_capacity_list),
    )

def plot_quantities(energy_levels, temperature_values, particle_count=1e5, save_path="Section5/B/"):
    beta_values = 1 / temperature_values
    chemical_potential, ground_state, d_ground_state_dT, heat_capacity = compute_quantities(
        energy_levels, beta_values, particle_count
    )

    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(5, sharex=True, figsize=(6, 10))

    axes[0].plot(temperature_values, -chemical_potential, label=r'$-\mu$')
    axes[0].set_ylabel(r'$-\mu$')
    axes[0].set_yscale('log')

    axes[1].plot(temperature_values, ground_state, label=r'$\langle n_0 \rangle$')
    axes[1].set_ylabel(r'$\langle n_0 \rangle$')

    axes[2].plot(temperature_values, ground_state, label=r'$\langle n_0 \rangle$')
    axes[2].set_ylabel(r'$\langle n_0 \rangle$')
    axes[2].set_yscale('log')

    axes[3].plot(temperature_values, -d_ground_state_dT, label=r'$-\frac{d\langle n_0 \rangle}{dT}$')
    axes[3].set_ylabel(r'$-\frac{d\langle n_0 \rangle}{dT}$')

    axes[4].plot(temperature_values, heat_capacity, label=r'$C_V$')
    axes[4].set_ylabel(r'$C_V$')

    axes[1].set_xscale('log')
    plt.xlabel(r'$T$')

    plt.tight_layout()

    save_filename = save_path + f"bose_system_analysis_T_{int(temperature_values[0])}_{int(temperature_values[-1])}.png"
    plt.savefig(save_filename, dpi=300)
    print(f"Plot saved to {save_filename}")

    plt.show()

plot_quantities(np.linspace(0, 1, 2), np.geomspace(1e1, 1e8, 100))
plot_quantities(np.linspace(0, 2000, 200)**(1/20), np.geomspace(1e-1, 1e6, 100))