# %% This defines a Jupyter-like code cell, while still being in a .py file
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

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

# Plotting the two solutions against each other

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

# %% BANDSTRUCTURE!!! (added in 2026)

# making a correspondence between the sorted energy eigenvalues and the eigenfunctions in the order they are produced in
energy_dict = []
for i in sorted_eigenvalues:
    energy_dict.append(list(eigenvalues).index(i))
energy_dict = np.array(energy_dict)

# These are just for reference, they are not used in the code
point_X = np.array([1],np.float64) # Point on the BZ boundary directly along the k_x direction in the reciprocal lattice 
point_Gamma = np.array([0],np.float64) # Origin in the reciprocal lattice

#eta = 2*np.pi*np.arange(L)/L # Not sure why I chose 0 to 2pi.     I did for the fourier transform, but it can be done either way
eta = np.pi*np.arange(-L,L,2)/L # excludes the upper end, includes lower end

index = np.arange(L) # for the 1D case

def Fourier_Transform(m):
    eigfunc = eigenvectors[:,m] # Loading the chosen wavefunction
    phi = [] # initialising the fourier transformed wavefunction
    for e in index: # this loops over the "k"s to get all terms of the FTed eigenfunction
        phi_e = 0
        for p in index:
            phi_e += np.exp(-1j*eta[e]*np.dot(point_X,np.array(p)))*eigfunc[p] # p here is the value of "x" along the lattice/chain.
        phi.append(np.square(np.abs(phi_e))) # I am saving the absolute value squared of each term of the fourier transform. This is because we need how big each "term" is.
        # The np.square or np.abs seems to return an array which requires me to reshape things later.
    return np.array(phi)

# works for this, but is more complex in 2D
def bandstructure():
    band = [] # initialising the bandstructure
    for n in index:
        band.append(Fourier_Transform(energy_dict[n])) # we are appending the entire fourier transformed wavefunction to bands here 
        # NOTE That is gonna be bad for the case where we only need parts of this for each direction.
        # Definitely should call each FT only once and use that judiciously to extract the info we want all at once.
    return np.array(band)

bds = bandstructure()

# %% plotting
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, int(L), dtype=float), sorted_eigenvalues) 
# Y has to be the sorted energy values otherwise, it plots everything "linearly". 
# The energy values for each wavefunction are not uniform and that is important.
Bands= np.reshape(bds,(100,100))

# NOTE To get the bandstructure for Gamma to X, reduce the range of X and reselect data from Bands be from 0 to pi

fig, ax = plt.subplots()

plt.ylabel("Energy")
plt.xlabel("k")
im = ax.pcolormesh(X, Y, Bands, cmap='plasma')
ax.set_title("Band structure")
fig.colorbar(im)
plt.show()
# %%
