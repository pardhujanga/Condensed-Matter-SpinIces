# %% This defines a Jupyter-like code cell, while still being in a .py file
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


# %% 
L=100 # Lattice size

# Kronecker delta function
def kron(x,y): 
    if x == y:
        return 1
    else:        
        return 0

t = np.ones(L) # Hopping parameter, chosen to be uniform
# the absolute value does not matter for this model

# Initialising the Hamiltonian
H = np.zeros((L,L))

# Building the Hamiltonian
for i in range(L):
    for j in range(L):
        H[i][j] = - kron(i,(j-1)%L) * t[(j-1)%L] - kron(i,(j+1)%L) * t[j]



# %% Solving the Hamiltonian 
eigenvalues, eigenvectors = linalg.eig(H)
sorted_eigenvalues = np.sort_complex(eigenvalues)

# Theoretical prediction for the energy spectrum
# Changed in 2026 for clarity
x = 2*np.pi*np.arange(1,L+1)/(L)
theory_energy_values = 2*np.cos(x)
sorted_theory_values = np.sort(theory_energy_values)


# %% Plotting the two solutions against each other

# Changing the render resolution to be able to see things more clearly
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.xlabel('Energy level, n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted_theory_values, 'k')
plt.scatter(range(L), sorted_eigenvalues, s=5, c="Red")
plt.show()

# As expected it is a sine curve. Computed the values of cos(k)
# which are the exact eigenvalues, with k being quantized. Match exactly.
# %%
