# %% Defining and solving the Hamiltonian for a 2D lattice in a uniform magnetic field perpendicular to it
from numba import jit
import numpy as np
import scipy.sparse as sp # Tried sparse matrix techniques, but my Hamiltonian was not sparse
from numpy import linalg #NOTE numba doesn't recognize scipy. can be made to if needed
from matplotlib import pyplot as plt

L = 20 # Size of the lattice

@jit(nopython = True) # Numba decorator to allow this function to be compiled "just in time" (jit)
def kron(x,y): # Kronecker delta function
    if x == y:
        return 1
    else:
        return 0


# Modified for clarity in 2026
index = np.arange(L**2)
length = np.arange(L)


@jit(nopython = True)
def energy(b):
    H = np.zeros((L**2,L**2),dtype=np.complex128) # intializing the Hamiltonian 
    for i in index:
        for x in length:
            for y in length:
                if x!=0 and x!=L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==0:
                    H[i][L*y + x] = - kron(i,L*y+(x+1))*np.exp(-1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
                elif x==L-1:
                    H[i][L*y + x] = - kron(i,L*y+(x-1))*np.exp(1j*b*y/2) - kron(i,L*((y-1))+x)*np.exp(-1j*b*x/2) - kron(i,L*((y+1))+x)*np.exp(1j*b*x/2)
    
    #NOTE Needed to split the hamiltoninan generation to avoid interactions between endpoints of consecutive "lines of atoms"
    # Removed Periodic boundary conditions as Hofstatder is meant to be on a normal finite lattice.
    # The periodicity arises naturally when we have rational flux ratio (all B in our case as we can only store rational numbers).

    # solving for spectrum and eigenfunctions
    # included in the Hamiltonian function as we call this several times in this script unlike in the other ones.
    return linalg.eig(H)
     

# Modified in 2026 for clarity
g = 400 # Gradations in the magnetic field
B = 2*np.pi*np.arange(g+1)/g # An array of magnetic field values (this is technically the flux ratio, but is physically the same idea)

# %%
eigenvalues,eigenvectors = energy(B[200])
sorted_eigenvalues = np.sort(eigenvalues)

# Creating the theoretical energy spectrum for the no-magnetic field case.
x = 2*np.pi*np.arange(1,L+1)/(L)
cos = np.cos(x)
# Modified for clarity in 2026
theory_energy_values = [2*i + 2*j for i in cos  for j in cos]
sorted_theory_values = np.sort(theory_energy_values)

# Changing the render resolution to be able to see things more clearly
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Plotting the two solutions against each other
plt.xlabel('Energy level, n')
plt.ylabel('Energy (in multiples of hopping parameter t)')
plt.plot(sorted_theory_values,'k')
plt.scatter(range(L**2), sorted_eigenvalues, s=1)
plt.show() 


# %% Producing Hofstadter's butterfly
# Rewritten for clarity in 2026
bands = np.zeros((L**2,g+1))
for p in np.arange(g+1):
    ener,level = energy(B[p]) 
    e = np.sort(ener)
    for q in np.arange(len(e)):
        bands[q][p]=e[q] 
        # each energy level's energy values at a given magnetic field are stored
        # in the corresponding 1D array within the overall 2D array.

# %% Plotting the Hofstadter's butterfly (Energy levels vs Magnetic field)
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500

for p in np.arange(L**2):
    plt.plot(np.linspace(0,2*np.pi, len(bands[p])), bands[p], 'k',linewidth = 0.1)

plt.xlabel('Flux Ratio')
plt.ylabel('Energy')
plt.show()


# %%  Choosing a mag field strength to find the expansion of eigenfunctions in the lattice basis
a = 100 # picking the strength of mag field
E,landau_levels = energy(B[a])

# %% Finding the lowest and highest energy eigenfunctions
print(f"The index of the lowest energy eigenfucntion is {np.argmin(E)}")
print(f"The index of the highest energy eigenfucntion is {np.argmax(E)}")

# %% 
n = 50 # picking the eigenfucntion to display (between 0 and L**2 -1 inclusive)

# %% plotting the amplitude of the eigenfunctions
Amplitude = np.abs(landau_levels[:,n]) # We are selecting all the numbers in the nth column. This is our nth eigenfunction in a flattened form.

# Remapping the eigenfunction to the lattice basis (reversing the linearisation)
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = Amplitude[L*Y + X] 

fig, ax = plt.subplots()

im_amp = ax.imshow(Z, cmap='plasma', vmin = 0, origin="lower")
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im_amp)
plt.show()

# %% plotting the phase of the eigenfunctions
Phase = np.angle(landau_levels[:,n])

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Zp = Phase[L*Y + X]

fig, ax = plt.subplots()

im_phase = ax.imshow(Zp, cmap='twilight_shifted', vmin = -3.14, vmax = 3.14, origin="lower") # This cmap has a cyclic colour scheme
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im_phase)
plt.show()

# %%
