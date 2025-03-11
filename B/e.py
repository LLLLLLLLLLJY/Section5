#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 100  # Total number of particles
beta_eps = np.linspace(0, 1, 100)  # Range of beta*epsilon

# Compute average particle numbers
n1 = 1 / (np.exp(beta_eps) - 1) - (1 + N) / (np.exp(beta_eps * (1 + N)) - 1)
n1[0] = N / 2  # Handling singularity at beta_eps = 0
n0 = N - n1  # Ground state occupancy

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(beta_eps, n0, label=r'$\langle n_0 \rangle$', color='blue')
plt.plot(beta_eps, n1, label=r'$\langle n_\epsilon \rangle$', color='red')

# Labels and title
plt.xlabel(r'$\beta\epsilon$', fontsize=12)
plt.ylabel('Average Particle Number', fontsize=12)
plt.title(f'Average Particle Number for $N={N}$', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot
plt.savefig("Section5/B/average_particle_number_quantum.png", dpi=300)

# Show plot
plt.show()