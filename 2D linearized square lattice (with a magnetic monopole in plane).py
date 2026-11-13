# %% Defining and solving the Hamiltonian for a 2D lattice with a magnetic monopole in it
from numba import jit
import numpy as np
import scipy.sparse as sp # Tried to use sparse matrices, but was not suitable (added comment in 2026)
from scipy import linalg
from matplotlib import pyplot as plt

L = 50 # Size of the lattice
# Modified in 2026 for clarity
index = np.arange(L**2)
length = np.arange(L)

# (a,b) is the monopole location. Choose (x-1/2, y-1/2) to place it at the centre of a unit cell.
a,b = L-1/2, L-1/2 

@jit(nopython = True)
def kron(x,y): # Kronecker Delta function
    if x == y:
        return 1
    else:
        return 0


# This is much faster at making the Hamiltonian
@jit(nopython = True)
def Hamiltonian():
    M = np.ones((L**2,L**2),dtype=np.complex128) #intializing the hamiltonian 

    for i in index:
        for x in length:
            for y in length:
                if x!=0 and x!=L-1:
                    M[i][L*y + x] *= - kron(i,L*y+(x-1))   * np.exp(0.5j*(np.arctan((x-a)/(y-b))   - np.arctan((x-1-a)/(y-b)))) \
                                     - kron(i,L*y+(x+1))   * np.exp(0.5j*(np.arctan((x-a)/(y-b))   - np.arctan((x+1-a)/(y-b)))) \
                                     - kron(i,L*((y-1))+x) * np.exp(0.5j*(np.arctan((y-1-b)/(x-a)) - np.arctan((y-b)/(x-a)))) \
                                     - kron(i,L*((y+1))+x) * np.exp(0.5j*(np.arctan((y+1-b)/(x-a)) - np.arctan((y-b)/(x-a))))
                elif x==0:
                    M[i][L*y + x] *= - kron(i,L*y+(x+1))   * np.exp(0.5j*(np.arctan((x-a)/(y-b))   - np.arctan((x+1-a)/(y-b)))) \
                                     - kron(i,L*((y-1))+x) * np.exp(0.5j*(np.arctan((y-1-b)/(x-a)) - np.arctan((y-b)/(x-a)))) \
                                     - kron(i,L*((y+1))+x) * np.exp(0.5j*(np.arctan((y+1-b)/(x-a)) - np.arctan((y-b)/(x-a))))
                elif x==L-1:
                    M[i][L*y + x] *= - kron(i,L*y+(x-1))   * np.exp(0.5j*(np.arctan((x-a)/(y-b))   - np.arctan((x-1-a)/(y-b)))) \
                                     - kron(i,L*((y-1))+x) * np.exp(0.5j*(np.arctan((y-1-b)/(x-a)) - np.arctan((y-b)/(x-a)))) \
                                     - kron(i,L*((y+1))+x) * np.exp(0.5j*(np.arctan((y+1-b)/(x-a)) - np.arctan((y-b)/(x-a))))
    return M

#NOTE Needed to split the hamiltoninan generation as the way the lattice to avoid interactions between
# endpoints of consecutive "lines of atoms".
H = Hamiltonian()

#NOTE: this is the bottle neck now. Making the Hamiltonian is very quick
#solving for spectrum and eigenfunctions
eigenvalues, eigenvectors = linalg.eig(H)
sorted_eigenvalues = np.sort(eigenvalues)

# %% Plotting the energy spectrum
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

#NOTE:The energy spectrum varies just a bit from the no magnetic field case.
# Could be due to the size of the lattice washing out the effect of the monopole.

 
# %% Finding the lowest and highest energy eigenfunctions
print(f"The index of the lowest energy eigenfucntion is {np.argmin(eigenvalues)}")
print(f"The index of the highest energy eigenfucntion is {np.argmax(eigenvalues)}")

# %% 
n = 59 # picking the eigenfucntion to display (between 0 and L**2-1 inclusive)

# %% plotting the amplitude of the eigenfunctions
Amplitude = np.abs(eigenvectors[:,n]) # We are selecting all the numbers in the nth column. This is our nth eigenfunction in a flattened form.

# Remapping the eigenfunction to the lattice basis (reversing the linearisation)
X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Z = Amplitude[L*Y + X] 

fig, ax = plt.subplots()

im_amp = ax.imshow(Z, cmap='plasma', origin="lower")
ax.set_title("absolute value of eigenfunction on the lattice")
fig.colorbar(im_amp)
plt.show()

# %% plotting the phase of the eigenfunctions
Phase = np.angle(eigenvectors[:,n])

X, Y = np.meshgrid(np.linspace(0,L-1,L,dtype=int), np.linspace(0,L-1,L,dtype=int))
Zp = Phase[L*Y + X]

fig, ax = plt.subplots()

im_phase = ax.imshow(Zp, cmap='twilight_shifted', vmin = -3.14, vmax = 3.14, origin="lower") # This cmap has a cyclic colour scheme
ax.set_title("phase of eigenfunction on the lattice")
fig.colorbar(im_phase)
plt.show()


# %%
