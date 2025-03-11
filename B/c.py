#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Define range of beta*epsilon
beta_eps = np.linspace(0, 5, 100)

# Compute average particle numbers in ground and excited states
n0 = 1 / (1 + np.exp(beta_eps))  # Ground state occupancy
n1 = 1 / (1 + np.exp(-beta_eps))  # Excited state occupancy

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(beta_eps, n0, label=r'$\langle n_0 \rangle_C / N$', color='blue')
plt.plot(beta_eps, n1, label=r'$\langle n_\epsilon \rangle_C / N$', color='red')

# Labels and title
plt.xlabel(r'$\beta\epsilon$', fontsize=12)
plt.ylabel('Fraction of Particles', fontsize=12)
plt.title('Average Particle Number in Canonical Ensemble', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot
plt.savefig("Section5/B/average_particle_number.png", dpi=300)

# Show plot
plt.show()